"use client";
import Loading from "@/components/Loading";
import Toast from "@/components/Toast";
import CampaignBook from "@/components/CampaignBook";

import { useEffect, useState } from "react";

export default function Home() {
    return (
        <div className="hero h-screen">
            <div className="hero-content text-center">
                <div className="max-w-md">
                    <h1 className="text-5xl font-bold obsidian-colour pb-16 select-none">Welcome back</h1>

                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-16 place-items-center">
                        <CampaignBook />
                    </div>

                </div>
            </div>
        </div>
    );
}


{/* <Toast type="error" message="Task failed successfully." />
<Toast message="New mail arrived." /> */}
