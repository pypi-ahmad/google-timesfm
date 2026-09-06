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
  const recoveryKey = `timesfm:draft:${context.workspace}:${context.draftId ?? `new:${context.versions.join(",")}`}`;
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
      if (inFlight.current || (!copy && state === "conflict")) return;
      inFlight.current = true;
      setState("saving");
      const submitted = structuredClone(latest.current);
      const previous = record.current;
      const savedKey = recoveryKey;
      try {
        const payload = { spec: submitted };
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
          setState(
            cause instanceof ApiError && cause.status === 409
              ? "conflict"
              : "error",
          );
          setError(cause instanceof Error ? cause : new Error(String(cause)));
        }
      } finally {
        inFlight.current = false;
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
    try {
      sessionStorage.setItem(
        recoveryKey,
        json({ spec: values, revision: record.current?.revision ?? null }),
      );
    } catch {
      /* Server persistence remains available when browser storage is full. */
    }
    if (state === "conflict" || state === "error" || inFlight.current) return;
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
