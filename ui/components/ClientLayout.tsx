"use client";

import { usePathname } from "next/navigation";
import NavBar from "@/components/navbar";
import { RecordingProvider } from "@/contexts/RecordingContext";
import RecordingPopup from "@/components/RecordingPopup";
import { CampaignProvider } from "@/contexts/CampaignContext";


export default function ClientLayout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const isMainPage = pathname === "/";

    return (
        <CampaignProvider>
            <RecordingProvider>
                {/* does not show navbar on main page */}
                {!isMainPage && <NavBar />}

                <main
                    className={`flex-1 flex flex-col items-center justify-center w-screen ${isMainPage ? "" : "pt-16"
                        } bg-[#eff1ed] overflow-hidden`}
                >
                    {children}
                </main>

                <RecordingPopup />
            </RecordingProvider>
        </CampaignProvider>
    );
}
