import type { ComponentProps, ReactNode } from "react";
import { AlertCircle, LoaderCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { errorMessage } from "@/lib/api";

export function Input({ className, ...props }: ComponentProps<"input">) {
  return <input className={cn("control", className)} {...props} />;
}
export function Select({ className, ...props }: ComponentProps<"select">) {
  return <select className={cn("control pe-7", className)} {...props} />;
}
export function Field({
  label,
  children,
  hint,
  className,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
  className?: string;
}) {
  return (
    <label className={cn("grid gap-2 text-sm font-medium", className)}>
      <span>{label}</span>
      {children}
      {hint && (
        <span className="text-caption font-normal text-muted-foreground">
          {hint}
        </span>
      )}
    </label>
  );
}
export function Check({
  label,
  hint,
  ...props
}: Omit<ComponentProps<"input">, "type"> & { label: string; hint?: string }) {
  return (
    <label className="flex items-start gap-3 py-2 text-sm leading-5">
      <input
        {...props}
        type="checkbox"
        className="mt-0.5 size-4 shrink-0 accent-primary"
      />
      <span>
        {label}
        {hint && (
          <span className="text-caption block text-muted-foreground">
            {hint}
          </span>
        )}
      </span>
    </label>
  );
}
export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "blue";
}) {
  return <span className={cn("badge", `badge-${tone}`)}>{children}</span>;
}
export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded-lg border border-destructive/25 bg-destructive/5 p-3 text-sm leading-5 text-destructive"
    >
      <AlertCircle className="mt-0.5 size-4 shrink-0" />
      <span>{errorMessage(error)}</span>
    </div>
  );
}
export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div
      role="status"
      className="flex items-center justify-center gap-2 p-8 text-sm text-muted-foreground"
    >
      <LoaderCircle className="size-4 animate-spin" />
      {label}
    </div>
  );
}
export function Empty({
  title,
  description,
  action,
  icon,
}: {
  title: string;
  description: string;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex min-h-56 flex-col items-center justify-center gap-3 px-8 py-10 text-center">
      {icon && (
        <div className="mb-1 flex size-12 items-center justify-center rounded-xl border bg-muted/50 text-muted-foreground">
          {icon}
        </div>
      )}
      <h3 className="text-section font-semibold">{title}</h3>
      <p className="text-caption max-w-sm text-muted-foreground">
        {description}
      </p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
export function PageHeading({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div>
        {eyebrow && (
          <p className="text-micro mb-2 font-semibold text-primary">
            {eyebrow}
          </p>
        )}
        <h1 className="text-title font-semibold">{title}</h1>
        <p className="text-caption mt-2 max-w-3xl text-pretty text-muted-foreground">
          {description}
        </p>
      </div>
      {actions && (
        <div className="flex flex-wrap items-center gap-2 pt-1">{actions}</div>
      )}
    </div>
  );
}
export function Section({
  title,
  description,
  children,
  actions,
  className,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("panel", className)}>
      <div className="flex items-start justify-between gap-3 border-b px-4 py-3">
        <div>
          <h2 className="text-section font-semibold">{title}</h2>
          {description && (
            <p className="text-caption mt-1 text-pretty text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}
