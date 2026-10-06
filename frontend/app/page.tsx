import { Suspense } from "react";
import LookupView from "./LookupView";

// useSearchParams in LookupView needs a Suspense boundary for static rendering.
export default function Page() {
  return (
    <Suspense>
      <LookupView />
    </Suspense>
  );
}
