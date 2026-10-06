import type { Metadata, Viewport } from "next";
import Sidebar from "@/components/Sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fantasy Projections",
  description: "Weekly PPR projections for QBs, RBs, WRs and TEs.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">
            <div className="content">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
