"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import {
  Activity,
  ArrowUpRight,
  Box,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  LineChart,
  Menu,
  Monitor,
  Moon,
  Radio,
  Sun,
  Waves,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { type ReactNode, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { cn, shortId } from "@/lib/utils";
import { useAnalyticalContext } from "@/hooks/use-context";
import { Badge } from "./ui/controls";
import { Button } from "./ui/button";
import { Drawer } from "./ui/dialog";
import { WorkspaceSettings } from "./workspace-settings";

const routes = [
  { path: "/overview", label: "Overview", icon: Gauge },
  { path: "/data", label: "Data", icon: Database },
  { path: "/forecasts", label: "Forecasts", icon: LineChart },
  { path: "/experiments", label: "Experiments", icon: FlaskConical },
  { path: "/scenarios", label: "Scenarios", icon: GitBranch },
  { path: "/tracking", label: "Tracking", icon: Radio },
  { path: "/models", label: "Models", icon: Box },
];
export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const context = useAnalyticalContext();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) =>
      api<Record<string, unknown>>("/health", { signal }),
    refetchInterval: 15_000,
    retry: false,
  });
  const nav = (
    <nav aria-label="Main navigation" className="space-y-1">
      {routes.map(({ path, label, icon: Icon }) => (
        <Link
          key={path}
          href={context.href(path)}
          aria-current={
            pathname === path || (pathname === "/" && path === "/overview")
              ? "page"
              : undefined
          }
          className={cn(
            "nav-link",
            (pathname === path || (pathname === "/" && path === "/overview")) &&
              "nav-link-active",
          )}
        >
          <Icon size={17} strokeWidth={1.7} />
          {label}
        </Link>
      ))}
    </nav>
  );
  return (
    <div className="min-h-screen">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar">
        <Link
          href={context.href("/overview")}
          className="flex items-center gap-2.5 px-2 py-4"
        >
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Waves size={21} />
          </span>
          <div>
            <span className="text-[15px] font-semibold tracking-tight">
              TimesFM
            </span>
            <span className="text-micro ms-1.5 font-medium text-muted-foreground">
              3.0
            </span>
            <p className="text-micro text-muted-foreground">
              Forecasting workspace
            </p>
          </div>
        </Link>
        <div className="my-5 flex min-h-11 items-center rounded-lg bg-muted px-3 py-2.5 text-sm">
          <span className="flex items-center gap-2">
            <Monitor size={14} className="text-muted-foreground" />
            Local workspace
          </span>
        </div>
        <p className="text-micro mb-2 px-3 font-semibold text-muted-foreground">
          Workspace
        </p>
        {nav}
        <div className="mt-auto space-y-3 border-t pt-4">
          <WorkspaceSettings />
          <div className="text-micro flex items-center gap-2 px-2 text-muted-foreground">
            <span
              className={cn(
                "size-1.5 rounded-full",
                health.isSuccess ? "bg-emerald-500" : "bg-amber-500",
              )}
            />
            {health.isSuccess
              ? "API connected"
              : health.isPending
                ? "Connecting to API"
                : "API unavailable"}
          </div>
          <a
            href="http://127.0.0.1:8501"
            target="_blank"
            rel="noreferrer"
            className="text-micro flex items-center gap-2 px-2 text-muted-foreground hover:text-foreground"
          >
            Streamlit diagnostics
            <ArrowUpRight size={12} />
          </a>
        </div>
      </aside>
      <div className="workspace-main">
        <header className="topbar">
          <div className="flex min-w-0 items-center gap-3">
            <div className="mobile-menu">
              <Drawer
                title="Workspace navigation"
                trigger={
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="Open navigation"
                  >
                    <Menu />
                  </Button>
                }
              >
                {nav}
                <div className="mt-5 border-t pt-4">
                  <WorkspaceSettings />
                </div>
              </Drawer>
            </div>
            <span className="hidden text-xs text-muted-foreground sm:inline">
              Workspace
            </span>
            <span className="hidden text-border sm:inline">/</span>
            <span className="text-xs font-medium">
              {routes.find((item) => item.path === pathname)?.label ??
                "Overview"}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Badge tone="blue">TimesFM-3</Badge>
            <label className="flex items-center gap-1.5 text-muted-foreground">
              <span className="sr-only">Color theme</span>
              {mounted && theme === "dark" ? (
                <Moon size={14} />
              ) : mounted && theme === "system" ? (
                <Monitor size={14} />
              ) : (
                <Sun size={14} />
              )}
              <select
                aria-label="Color theme"
                className="bg-transparent text-sm outline-none focus-visible:ring-2"
                value={mounted ? theme : "light"}
                onChange={(event) => setTheme(event.target.value)}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </label>
          </div>
        </header>
        {(context.versions.length > 0 || context.runId || context.draftId) && (
          <div className="context-bar">
            <Activity size={13} />
            <span className="font-medium">Analytical context</span>
            {context.versions.length > 0 && (
              <Link href={context.href("/data")} className="context-chip">
                {context.versions.length === 1
                  ? `Version ${shortId(context.versions[0])}`
                  : `${context.versions.length} dataset versions`}
              </Link>
            )}
            {context.draftId && (
              <Link href={context.href("/forecasts")} className="context-chip">
                Draft {shortId(context.draftId)}
              </Link>
            )}
            {context.runId && (
              <Link href={context.href("/forecasts")} className="context-chip">
                Run {shortId(context.runId)}
              </Link>
            )}
            {context.target && (
              <span className="context-chip">{context.target}</span>
            )}
          </div>
        )}
        <main id="main" className="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}
