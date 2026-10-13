import type { Metadata } from "next";

import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "Recut - long videos into shorts, captions, chapters and posts",
  description: "Turn a long video or podcast into vertical clips with animated captions, YouTube chapters, show notes, a blog post and social posts. Every output is reviewed before it is published.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
