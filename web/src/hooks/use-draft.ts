"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { api, ApiError, json } from "@/lib/api";
import { defaultSpec, specSchema, type Spec } from "@/lib/spec";
import type { RecordItem } from "@/lib/types";
import { useAnalyticalContext } from "./use-context";

type Draft = RecordItem<{ spec: Spec }>;
type SaveState = "saved" | "saving" | "unsaved" | "conflict" | "error";
type Recovery = { spec: Spec; revision: number | null };

// Autosaving react-hook-form binding for the forecast Spec: debounces
// edits into a PATCH/POST to the drafts API, mirrors unsaved edits into
// sessionStorage for crash/reload recovery, and surfaces optimistic-
// concurrency conflicts (another tab saved the same draft) as a distinct
// state rather than silently overwriting. Sole consumer is
// features/forecasts-page.tsx. See lib/spec.ts for the Spec shape and
// hooks/use-context.ts for how draftId/versions drive which draft loads.
export function useDraft() {
  const context = useAnalyticalContext();
  const client = useQueryClient();
  const form = useForm<Spec>({
    resolver: zodResolver(specSchema),
    defaultValues: defaultSpec(context.versions),
  });
  const values = useWatch({ control: form.control }) as Spec;
  const snapshot = JSON.stringify(values);
  const [state, setState] = useState<SaveState>("saved");
  const [error, setError] = useState<Error | null>(null);
  const [epoch, setEpoch] = useState(0);
  const record = useRef<Draft | null>(null);
  const savedSnapshot = useRef(snapshot);
  const latest = useRef(values);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const loadedRecovery = useRef("");
  // Keyed by draft id, or by the pending dataset-version selection when
  // there's no draft yet — so an unsaved "new draft" recovers correctly
  // even before it has a server id.
  const recoveryKey = `timesfm:draft:${context.workspace}:${context.draftId ?? `new:${context.versions.join(",")}`}`;
  // Snapshot of the key a save was started under, checked after the async
  // save resolves: if the user has since switched to a different draft
  // (context.draftId changed), that save's result must not be applied to
  // the now-current draft state — this guards the race between navigating
  // away and an in-flight save/load resolving late.
  const activeRecoveryKey = useRef(recoveryKey);
  activeRecoveryKey.current = recoveryKey;
  latest.current = values;
  const selected = useQuery({
    queryKey: ["draft", context.draftId],
    queryFn: ({ signal }) =>
      api<Draft>(`/drafts/${context.draftId}`, { signal }),
    enabled: !!context.draftId,
  });
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    // Wait for the server draft (if any) to actually load before deciding
    // what to restore — loading a stale/empty state here would otherwise
    // race the query and momentarily show the wrong spec.
    if (
      context.draftId &&
      (!selected.data || selected.data.id !== context.draftId)
    )
      return;
    if (loadedRecovery.current === recoveryKey) return;
    loadedRecovery.current = recoveryKey;
    const server = selected.data;
    record.current = server ?? null;
    let recovered: Recovery | null = null;
    try {
      recovered = JSON.parse(sessionStorage.getItem(recoveryKey) ?? "null");
    } catch {
      /* A malformed local recovery must not replace a server draft. */
    }
    const initial = server?.payload.spec ?? defaultSpec(context.versions);
    savedSnapshot.current = JSON.stringify(initial);
    if (recovered?.spec) {
      form.reset(recovered.spec);
      // A locally recovered draft only conflicts with the server if it was
      // captured against a different revision AND actually differs from
      // what the server now has — recovering the same content the server
      // already saved is not a conflict.
      if (
        server &&
        recovered.revision !== server.revision &&
        JSON.stringify(recovered.spec) !== savedSnapshot.current
      )
        setState("conflict");
      else
        setState(
          JSON.stringify(recovered.spec) === savedSnapshot.current
            ? "saved"
            : "unsaved",
        );
    } else {
      form.reset(initial);
      setState("saved");
    }
    setError(null);
    if (
      server &&
      initial.dataset_version_ids.join(",") !== context.versions.join(",")
    )
      context.update({
        versions: initial.dataset_version_ids.join(","),
        version: null,
      });
  }, [context.draftId, context.versions, form, recoveryKey, selected.data]);

  const save = useCallback(
    async (copy = false) => {
      // A conflict blocks further silent autosaves until the user
      // explicitly resolves it (save as copy, or reload the server draft)
      // — otherwise a background save could clobber the other tab's write.
      if (inFlight.current || (!copy && state === "conflict")) return;
      inFlight.current = true;
      setState("saving");
      const submitted = structuredClone(latest.current);
      const previous = record.current;
      const savedKey = recoveryKey;
      try {
        const payload = { spec: submitted };
        // If-Match with the last-known revision makes this an optimistic-
        // concurrency update: the server rejects with 409 (caught below)
        // if another writer has since changed the draft.
        const result =
          previous && !copy
            ? await api<Draft>(`/drafts/${previous.id}`, {
                method: "PATCH",
                headers: { "If-Match": String(previous.revision) },
                body: json({ payload, revision: previous.revision }),
              })
            : await api<Draft>("/drafts", {
                method: "POST",
                body: json({
                  workspace_id: context.workspace,
                  name: copy
                    ? `${previous?.name ?? "Forecast"} · recovered copy`
                    : `Forecast · ${new Date().toLocaleDateString()}`,
                  payload,
                }),
              });
        sessionStorage.removeItem(savedKey);
        client.setQueryData(["draft", result.id], result);
        void client.invalidateQueries({ queryKey: ["drafts"] });
        // Only apply this save's result if the component is still mounted
        // and the user hasn't switched to a different draft while the
        // request was in flight (see activeRecoveryKey above).
        if (mounted.current && activeRecoveryKey.current === savedKey) {
          record.current = result;
          savedSnapshot.current = JSON.stringify(submitted);
          setState(
            JSON.stringify(latest.current) === savedSnapshot.current
              ? "saved"
              : "unsaved",
          );
          setError(null);
          if (result.id !== context.draftId) {
            const newKey = `timesfm:draft:${context.workspace}:${result.id}`;
            sessionStorage.setItem(
              newKey,
              json({ spec: latest.current, revision: result.revision }),
            );
            loadedRecovery.current = newKey;
            context.update({
              draft: result.id,
              versions: submitted.dataset_version_ids.join(","),
              version: null,
            });
          }
        }
      } catch (cause) {
        if (mounted.current && activeRecoveryKey.current === savedKey) {
          // A 409 means another writer changed the draft since our last
          // known revision — surfaced as "conflict", distinct from other
          // failures which are recoverable by retrying the same save.
          setState(
            cause instanceof ApiError && cause.status === 409
              ? "conflict"
              : "error",
          );
          setError(cause instanceof Error ? cause : new Error(String(cause)));
        }
      } finally {
        inFlight.current = false;
        // Bumping epoch re-triggers the autosave effect below even when
        // `snapshot` hasn't changed, so a save that finishes with
        // unsaved edits still queued (edited again mid-save) gets a fresh
        // debounce timer instead of being left unscheduled.
        if (mounted.current) setEpoch((value) => value + 1);
      }
    },
    [client, context, recoveryKey, state],
  );

  useEffect(() => {
    if (
      loadedRecovery.current !== recoveryKey ||
      snapshot === savedSnapshot.current
    )
      return;
    // Mirror every edit into sessionStorage immediately (ahead of the
    // debounce) so a reload or crash before the debounced save fires can
    // still recover the latest keystrokes.
    try {
      sessionStorage.setItem(
        recoveryKey,
        json({ spec: values, revision: record.current?.revision ?? null }),
      );
    } catch {
      /* Server persistence remains available when browser storage is full. */
    }
    if (state === "conflict" || state === "error" || inFlight.current) return;
    // Debounced autosave: waits 900ms of no further edits before writing
    // to the server draft.
    const timer = setTimeout(() => {
      void save();
    }, 900);
    return () => clearTimeout(timer);
  }, [snapshot, values, recoveryKey, save, state, epoch]);

  const reload = async () => {
    if (!context.draftId) return;
    const fresh = await api<Draft>(`/drafts/${context.draftId}`);
    record.current = fresh;
    savedSnapshot.current = JSON.stringify(fresh.payload.spec);
    form.reset(fresh.payload.spec);
    sessionStorage.removeItem(recoveryKey);
    setState("saved");
    setError(null);
    client.setQueryData(["draft", fresh.id], fresh);
  };
  const actualState =
    state === "saved" && snapshot !== savedSnapshot.current ? "unsaved" : state;
  return {
    form,
    values,
    state: actualState,
    error: error ?? selected.error,
    save,
    reload,
    isLoading: !!context.draftId && selected.isPending,
    draft: record.current,
  };
}
