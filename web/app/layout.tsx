import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// The fonts ship with the app (app/fonts) so a build never depends on Google Fonts being reachable.
const bebasNeue = localFont({
  src: [{ path: "./fonts/bebas-neue-latin-400-normal.woff2", weight: "400", style: "normal" }],
  variable: "--font-bebas",
  display: "swap",
});

const dmSans = localFont({
  src: [
    { path: "./fonts/dm-sans-latin-400-normal.woff2", weight: "400", style: "normal" },
    { path: "./fonts/dm-sans-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/dm-sans-latin-600-normal.woff2", weight: "600", style: "normal" },
  ],
  variable: "--font-dm-sans",
  display: "swap",
});

// Handwriting for sketch labels (training scenes).
const hand = localFont({
  src: [
    { path: "./fonts/caveat-latin-500-normal.woff2", weight: "500", style: "normal" },
    { path: "./fonts/caveat-latin-700-normal.woff2", weight: "700", style: "normal" },
  ],
  variable: "--font-hand",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ASN Pfeil Phönix — Spielanalyse",
  description: "Videoannotation und Spielanalyse für ASN Pfeil Phönix Nürnberg",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="de"
      className={`${bebasNeue.variable} ${dmSans.variable} ${hand.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#F8F8F6]">
        {children}
      </body>
    </html>
  );
}
