import type { Metadata } from "next";
import { Geist_Mono, Pirata_One, Noto_Serif, Faculty_Glyphic, Metal_Mania } from "next/font/google";

import "./globals.css";
import NavBar from "@/components/navbar";
import { RecordingProvider } from "@/contexts/RecordingContext";
import RecordingPopup from "@/components/RecordingPopup"

const pirataOne = Pirata_One({
  variable: "--font-pirata-one",
  subsets: ['latin'],
  weight: '400',
})

const metalMania = Metal_Mania({
  variable: "--font-metal-mania",
  subsets: ['latin'],
  weight: "400"
})

const notoSerif = Noto_Serif({
  variable: "--font-noto-serif",
  subsets: ["latin"],
});

const facultyGlyphic = Faculty_Glyphic({
  variable: "--font-faculty-glyphic",
  subsets: ["latin"],
  weight: "400",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Dungeon Scribe",
  description: "DECO3801 Sem 2 2025 Critical Fail",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${notoSerif.variable} ${pirataOne.variable} ${facultyGlyphic.variable} ${metalMania.variable} antialiased flex flex-col min-h-screen`}
        style={{ fontFamily: "var(--font-faculty-glyphic), serif" }}
      >
        <RecordingProvider>
          <NavBar />
          <main className="flex-1 flex flex-col items-center justify-center w-screen pt-16">
            {children}
          </main>
          <RecordingPopup />
        </RecordingProvider>
      </body>
    </html >
  );
}
