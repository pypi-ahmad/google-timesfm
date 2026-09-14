import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Small display/formatting helpers shared by tables, charts, and cards.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
export function shortId(value: string) {
  return value.slice(0, 8);
}
export function display(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number")
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, { maximumFractionDigits: 5 });
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
export function dateLabel(value: string) {
  // Renders in the viewer's local timezone via toLocaleString, not the
  // timestamp's original offset. Falls back to the raw string for
  // unparsable input rather than showing "Invalid Date".
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
}
