import { notFound } from "next/navigation";
import { DataPage } from "@/features/data-page";
import { ForecastsPage } from "@/features/forecasts-page";
import { ModelsPage } from "@/features/models-page";
import { OverviewPage } from "@/features/overview-page";
import { TrackingPage } from "@/features/tracking-page";

// Catch-all dynamic route mapping a URL page segment to a feature page
// component; all segments are statically enumerated via
// generateStaticParams. app/page.tsx handles the "/" root separately by
// rendering OverviewPage directly. See features/*.tsx for the pages
// themselves.
export function generateStaticParams() {
  return [
    "overview",
    "data",
    "forecasts",
    "experiments",
    "scenarios",
    "tracking",
    "models",
  ].map((page) => ({ page }));
}
export default async function Page({
  params,
}: {
  params: Promise<{ page: string }>;
}) {
  const { page } = await params;
  switch (page) {
    case "overview":
      return <OverviewPage />;
    case "data":
      return <DataPage />;
    case "forecasts":
      return <ForecastsPage />;
    case "experiments":
      return <ForecastsPage mode="experiment" />;
    case "scenarios":
      return <ForecastsPage mode="scenario" />;
    case "tracking":
      return <TrackingPage />;
    case "models":
      return <ModelsPage />;
    default:
      notFound();
  }
}
