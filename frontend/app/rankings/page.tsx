import type { Metadata } from "next";
import { Suspense } from "react";
import RankingsView from "./RankingsView";

export const metadata: Metadata = { title: "Rankings · Fantasy Projections" };

export default function Page() {
  return (
    <Suspense>
      <RankingsView />
    </Suspense>
  );
}
