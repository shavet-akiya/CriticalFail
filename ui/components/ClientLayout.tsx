"use client";

import { usePathname } from "next/navigation";
import NavBar from "@/components/navbar";
import { RecordingProvider } from "@/contexts/RecordingContext";
import RecordingPopup from "@/components/RecordingPopup";
import { CampaignProvider, useCampaign } from "@/contexts/CampaignContext";
import { useEffect, useState } from "react";

function LayoutContent({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const { selectedCampaign } = useCampaign();
    const [navState, setNavState] = useState<"none" | "blank" | "full">("none");

    useEffect(() => {
        if (pathname === "/" || pathname === "/new_campaign") {
            setNavState("none");
            return;
        }

        if (pathname.startsWith("/campaign")) {
            const parts = pathname.split("/").filter(Boolean);

            // /campaign/ or /campaign/id/
            if (parts.length === 1 || parts.length === 2) {
                setNavState("blank");
                return;
            }

            if (parts.length > 2) {
                setNavState("full");
            }
            return;
        }

        setNavState("blank");
    }, [pathname, selectedCampaign?.campaign_id]);

    return (
        <>
            {navState === "full" && <NavBar found />}
            {navState === "blank" && <NavBar found={false} />}

            <main
                className={`flex-1 flex flex-col items-center justify-center w-screen ${navState === "none" ? "" : "pt-16"
                    } bg-white-colour overflow-hidden`}
            >
                {children}
            </main>

            <RecordingPopup />
        </>
    );
}

export default function ClientLayout({ children }: { children: React.ReactNode }) {
    return (
        <CampaignProvider>
            <RecordingProvider>
                <LayoutContent>{children}</LayoutContent>
            </RecordingProvider>
        </CampaignProvider>
    );
}
