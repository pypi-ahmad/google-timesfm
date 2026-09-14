"use client";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  flexRender,
  createSortedRowModel,
  rowSortingFeature,
  sortFns,
  tableFeatures,
  useTable,
  type SortingState,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { api, query } from "@/lib/api";
import { display } from "@/lib/utils";
import type { Row, TableData } from "@/lib/types";
import { Button } from "./ui/button";
import { Empty, ErrorNotice, Loading } from "./ui/controls";

// Two related exports: DataTable renders a sortable, row-virtualized table
// over an in-memory Row[] (used for previews and small result sets), and
// RemoteTable wraps it with server-side pagination/sorting for a saved
// run's result tables (see lib/types.ts TableData, lib/api.ts).
const features = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns,
});
export function DataTable({
  data,
  columns: names,
  sorting: externalSorting,
  onSortingChange,
  caption,
}: {
  data: Row[];
  columns?: string[];
  sorting?: SortingState;
  onSortingChange?: (sorting: SortingState) => void;
  caption?: string;
}) {
  const id = useId();
  const [localSorting, setLocalSorting] = useState<SortingState>([]);
  const columns = useMemo(
    () =>
      (names ?? Object.keys(data[0] ?? {})).map((name) => ({
        id: name,
        accessorFn: (row: Row) => row[name],
        header: name.replaceAll("_", " "),
        cell: ({ getValue }: { getValue: () => unknown }) =>
          display(getValue()),
      })),
    [names, data],
  );
  const sorting = externalSorting ?? localSorting;
  const table = useTable({
    key: id,
    features,
    data,
    columns,
    state: { sorting },
    onSortingChange: (updater) => {
      const next = typeof updater === "function" ? updater(sorting) : updater;
      if (onSortingChange) onSortingChange(next);
      else setLocalSorting(next);
    },
    manualSorting: !!onSortingChange,
  });
  const scroll = useRef<HTMLDivElement>(null);
  // Tracks whether the table is wider than its scroll container and
  // whether there's more content past the visible end, to show a fade +
  // "scroll sideways" hint (app/globals.css .table-more-end/.table-scroll-
  // hint) — purely a discoverability affordance for wide tables.
  const [horizontalOverflow, setHorizontalOverflow] = useState({
    overflowing: false,
    moreAtEnd: false,
  });
  useEffect(() => {
    const element = scroll.current;
    if (!element) return;
    const update = () => {
      const remaining =
        element.scrollWidth -
        element.clientWidth -
        Math.abs(element.scrollLeft);
      setHorizontalOverflow({
        overflowing: element.scrollWidth > element.clientWidth + 1,
        moreAtEnd: remaining > 1,
      });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    element.addEventListener("scroll", update, { passive: true });
    return () => {
      observer.disconnect();
      element.removeEventListener("scroll", update);
    };
  }, [data, columns]);
  const rows = table.getRowModel().rows;
  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scroll.current,
    estimateSize: () => 44,
    overscan: 12,
  });
  // Only the visible row window is rendered; `top`/`bottom` size spacer
  // rows that stand in for the rest of the table's height so scrolling
  // and native table semantics (sticky header, row count) stay correct.
  const items = virtual.getVirtualItems();
  const top = items[0]?.start ?? 0;
  const bottom = Math.max(0, virtual.getTotalSize() - (items.at(-1)?.end ?? 0));
  if (!data.length)
    return (
      <Empty
        title="No rows to show"
        description="Results will appear here when this table contains matching data."
      />
    );
  return (
    <div
      className={`table-shell ${horizontalOverflow.moreAtEnd ? "table-more-end" : ""}`}
    >
      <div
        ref={scroll}
        className="table-scroller"
        tabIndex={0}
        role="region"
        aria-label={caption ?? "Data table"}
      >
        <table className="data-table" aria-rowcount={data.length + 1}>
          <caption className="sr-only">{caption ?? "Data table"}</caption>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th
                    key={header.id}
                    scope="col"
                    aria-sort={
                      header.column.getIsSorted() === "asc"
                        ? "ascending"
                        : header.column.getIsSorted() === "desc"
                          ? "descending"
                          : "none"
                    }
                  >
                    <button
                      type="button"
                      className="flex min-h-8 items-center gap-3"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                      {header.column.getIsSorted() === "asc" ? (
                        <ArrowUp size={11} />
                      ) : header.column.getIsSorted() === "desc" ? (
                        <ArrowDown size={11} />
                      ) : (
                        <ArrowUpDown size={10} className="opacity-40" />
                      )}
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {top > 0 && (
              <tr aria-hidden="true">
                <td
                  colSpan={columns.length}
                  style={{ height: top, padding: 0 }}
                />
              </tr>
            )}
            {items.map((item) => {
              const row = rows[item.index];
              return (
                <tr key={row.id} aria-rowindex={item.index + 2}>
                  {row.getAllCells().map((cell) => (
                    <td
                      key={cell.id}
                      title={display(cell.getValue())}
                      className={
                        typeof cell.getValue() === "number"
                          ? "text-end"
                          : undefined
                      }
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
            {bottom > 0 && (
              <tr aria-hidden="true">
                <td
                  colSpan={columns.length}
                  style={{ height: bottom, padding: 0 }}
                />
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {horizontalOverflow.overflowing && horizontalOverflow.moreAtEnd && (
        <div className="table-scroll-hint" aria-hidden="true">
          Scroll sideways to view all columns
        </div>
      )}
    </div>
  );
}

export function RemoteTable({
  runId,
  tableName,
  filters,
}: {
  runId: string;
  tableName: string;
  filters: Record<string, string>;
}) {
  // `sorting`/`offset` here are the current server-side request params
  // (the server does the actual sort/paging); passed down as
  // DataTable's externally-controlled sorting so DataTable never sorts
  // its own already-server-sorted rows.
  const [offset, setOffset] = useState(0);
  const [sorting, setSorting] = useState<SortingState>([]);
  const limit = 100;
  const params = {
    ...filters,
    offset,
    limit,
    sort_by: sorting[0]?.id,
    descending: sorting[0]?.desc,
  };
  const result = useQuery({
    queryKey: ["run-table", runId, tableName, params],
    queryFn: ({ signal }) =>
      api<TableData>(
        `/runs/${encodeURIComponent(runId)}/tables/${encodeURIComponent(tableName)}?${query(params)}`,
        { signal },
      ),
  });
  return (
    <div>
      <ErrorNotice error={result.error} />
      {result.isPending ? (
        <Loading label="Loading rows…" />
      ) : (
        result.data && (
          <DataTable
            data={result.data.rows}
            columns={result.data.columns}
            sorting={sorting}
            onSortingChange={(next) => {
              setSorting(next);
              setOffset(0);
            }}
            caption={`${tableName.replaceAll("_", " ")} results`}
          />
        )
      )}
      <div className="text-micro flex items-center justify-between gap-3 border-t px-4 py-2 text-muted-foreground">
        <span className="tabular">
          {result.data
            ? `${result.data.total === 0 ? 0 : offset + 1}–${Math.min(offset + limit, result.data.total)} of ${result.data.total.toLocaleString()} rows`
            : "100 rows per page"}
        </span>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Previous page"
            disabled={offset === 0 || result.isPending}
            onClick={() => setOffset(Math.max(0, offset - limit))}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Next page"
            disabled={
              !result.data ||
              offset + limit >= result.data.total ||
              result.isPending
            }
            onClick={() => setOffset(offset + limit)}
          >
            <ChevronRight />
          </Button>
        </div>
      </div>
    </div>
  );
}
