"use client";
import { useRef, useState } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  Database,
  FileSpreadsheet,
  FlaskConical,
  Plus,
  Upload,
} from "lucide-react";
import { api, json } from "@/lib/api";
import { dateLabel, display, shortId } from "@/lib/utils";
import type { DatasetVersion, TableData } from "@/lib/types";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useDatasets } from "@/hooks/use-records";
import { DataTable } from "@/components/data-table";
import { Button } from "@/components/ui/button";
import {
  Badge,
  Empty,
  ErrorNotice,
  Field,
  Input,
  Loading,
  PageHeading,
  Section,
  Select,
} from "@/components/ui/controls";

// Dataset library page: upload/version a CSV or Parquet file (or load the
// bundled demo dataset), browse existing versions, and preview a
// selected version's rows. Each upload is an immutable new "version" —
// see hooks/use-records.ts useDatasets and lib/types.ts DatasetVersion.
export function DataPage() {
  const context = useAnalyticalContext();
  const client = useQueryClient();
  const library = useDatasets();
  const fileInput = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const selected = library.versions.find(
    (item) => item.id === context.versions[0],
  );
  const preview = useQuery({
    queryKey: ["dataset-preview", selected?.id],
    queryFn: ({ signal }) =>
      api<TableData>(`/datasets/versions/${selected!.id}/preview`, { signal }),
    enabled: !!selected,
  });
  // One mutation covers both paths: loading the bundled demo dataset (no
  // file needed) and uploading a real file. FormData is used for the real
  // upload so lib/api.ts's fetch wrapper skips setting a JSON
  // Content-Type and lets the browser set the multipart boundary itself.
  const upload = useMutation({
    mutationFn: async (demo: boolean) => {
      if (demo)
        return api<DatasetVersion>(
          `/datasets/demo?workspace_id=${encodeURIComponent(context.workspace)}`,
          { method: "POST" },
        );
      if (!file) throw new Error("Choose a CSV or Parquet file first.");
      const body = new FormData();
      body.set("file", file);
      body.set("name", name || file.name);
      body.set("workspace_id", context.workspace);
      if (datasetId) body.set("dataset_id", datasetId);
      return api<DatasetVersion>("/datasets", { method: "POST", body });
    },
    onSuccess: (result) => {
      void client.invalidateQueries({ queryKey: ["datasets"] });
      void client.invalidateQueries({ queryKey: ["versions"] });
      context.update({ versions: result.id, version: null, draft: null });
      setFile(null);
      setName("");
      if (fileInput.current) fileInput.current.value = "";
    },
  });
  const chooseFile = (chosen?: File) => {
    if (chosen) {
      setFile(chosen);
      if (!name) setName(chosen.name.replace(/\.[^.]+$/, ""));
    }
  };
  return (
    <>
      <PageHeading
        title="Data library"
        description="Version your source data once. Reuse the same inputs across forecasts and experiments."
        actions={
          <Button
            variant="outline"
            disabled={upload.isPending}
            onClick={() => upload.mutate(true)}
          >
            <FlaskConical />
            Load demo data
          </Button>
        }
      />
      <div className="grid items-start gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        <Section
          title="Add a dataset"
          description="CSV or Parquet · each upload becomes a fixed version"
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              upload.mutate(false);
            }}
            className="space-y-4 p-4"
          >
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                chooseFile(event.dataTransfer.files[0]);
              }}
              className={`rounded-lg border border-dashed px-5 py-7 text-center ${dragging ? "border-primary bg-primary/5" : "border-input bg-muted/25"}`}
            >
              <Upload className="mx-auto mb-3 size-6 text-muted-foreground" />
              <p className="mb-1 text-xs font-medium">
                {file ? file.name : "Drop a dataset here"}
              </p>
              <p className="mb-3 text-[11px] text-muted-foreground">
                {file
                  ? `${(file.size / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} KB ready to upload`
                  : "or choose a file from your computer"}
              </p>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fileInput.current?.click()}
              >
                Browse files
              </Button>
              <input
                ref={fileInput}
                type="file"
                accept=".csv,.parquet"
                aria-label="Dataset file"
                className="sr-only"
                onChange={(event) => chooseFile(event.target.files?.[0])}
              />
            </div>
            <Field label="Dataset name">
              <Input
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Daily store demand"
              />
            </Field>
            <Field label="Version of">
              <Select
                value={datasetId}
                onChange={(event) => setDatasetId(event.target.value)}
              >
                <option value="">Create a new dataset</option>
                {library.datasets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </Select>
            </Field>
            <ErrorNotice error={upload.error} />
            <Button
              type="submit"
              className="w-full"
              disabled={!file || upload.isPending}
            >
              <Plus />
              {upload.isPending
                ? "Uploading…"
                : datasetId
                  ? "Upload new version"
                  : "Add dataset"}
            </Button>
          </form>
        </Section>
        <div className="space-y-5">
          <Section
            title="Available versions"
            description={`${library.datasets.length} datasets · ${library.versions.length} versions`}
          >
            <ErrorNotice error={library.error} />
            {library.isLoading ? (
              <Loading />
            ) : !library.versions.length ? (
              <Empty
                icon={<Database size={22} />}
                title="Build your data library"
                description="Upload your first dataset or load the included demo to explore the complete forecasting workflow."
              />
            ) : (
              <div className="max-h-96 overflow-auto divide-y">
                {library.versions.map((item) => (
                  <button
                    key={item.id}
                    className={`flex w-full items-center gap-3 px-4 py-3 text-start hover:bg-muted/60 ${selected?.id === item.id ? "bg-primary/5" : ""}`}
                    onClick={() =>
                      context.update({
                        versions: item.id,
                        version: null,
                        draft: null,
                      })
                    }
                  >
                    <div className="rounded-lg border bg-card p-2 text-primary">
                      <FileSpreadsheet size={19} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium">
                        {item.name || item.payload.filename}
                      </p>
                      <p className="mt-1 text-[10px] text-muted-foreground">
                        {display(item.payload.rows)} rows ·{" "}
                        {item.payload.columns?.length ?? 0} columns ·{" "}
                        {dateLabel(item.created_at)}
                      </p>
                    </div>
                    <Badge>{shortId(item.id)}</Badge>
                    <ArrowRight size={14} className="text-muted-foreground" />
                  </button>
                ))}
              </div>
            )}
          </Section>
          {selected && (
            <Section
              title={selected.name || "Dataset preview"}
              description={`${selected.payload.filename} · ${display(selected.payload.rows)} total rows · fixed version ${shortId(selected.id)}`}
              actions={
                <Button size="sm" asChild>
                  <Link
                    href={context.href("/forecasts", {
                      versions: selected.id,
                      draft: null,
                    })}
                  >
                    Use in forecast
                    <ArrowRight />
                  </Link>
                </Button>
              }
            >
              <ErrorNotice error={preview.error} />
              {preview.isPending ? (
                <Loading />
              ) : (
                preview.data && (
                  <>
                    <DataTable
                      data={preview.data.rows}
                      columns={preview.data.columns}
                      caption="Dataset preview"
                    />
                    <p className="border-t px-4 py-2 text-[10px] text-muted-foreground">
                      Previewing {preview.data.rows.length.toLocaleString()} of{" "}
                      {preview.data.total.toLocaleString()} rows. Full data
                      stays on the server.
                    </p>
                  </>
                )
              )}
              <details className="border-t p-4 text-[11px]">
                <summary className="cursor-pointer text-muted-foreground">
                  Version provenance
                </summary>
                <dl className="mt-3 space-y-2">
                  <div>
                    <dt className="text-muted-foreground">Source</dt>
                    <dd>{selected.payload.source_name}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">SHA-256</dt>
                    <dd className="break-all font-mono text-[10px]">
                      {selected.payload.artifact?.sha256}
                    </dd>
                  </div>
                </dl>
              </details>
            </Section>
          )}
        </div>
      </div>
    </>
  );
}
