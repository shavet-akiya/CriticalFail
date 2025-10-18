// src/App.jsx
import React from "react";
import Link from "next/link";
import CamapignCard from "@/components/CampaignCard";

// future consideration: if campaign id: show welcome back
function App() {
    return (
        <>
            <div className="h-screen w-full overflow-y-scroll snap-y snap-mandatory scroll-smooth">
                <section className="h-screen flex flex-col items-center justify-center bg-[#eff1ed] text-[#3c1642] snap-start relative gap-32">
                    <h1 className="text-9xl font-bold select-none font-metal-mania red-colour text-center text-shadow-lg text-shadow-gray-300">
                        Dungeon Scribe
                    </h1>

                    {/* Scroll hint */}
                    <div className="bottom-20 text-center flex flex-row gap-8">
                        <a href="#campaign_selection" className="btn btn-primary scroll-smooth">
                            Select Campaign
                        </a>

                        <button className="btn btn-primary">
                            <Link href={`/new_campaign`}>New Campaign</Link>
                        </button>
                    </div>
                </section>

                {/* Section 2 Select campaign */}
                <section
                    id="campaign_selection"
                    className="flex flex-col items-center justify-center bg-[#e0d6cb] text-[#3c1642] snap-start"
                >
                    <h2 className="text-4xl font-bold mb-4 pt-16">Campaign Selection</h2>

                    {/* <p className="text-lg max-w-xl text-center">
                        This is where your available campaigns will appear. You can scroll or click above to return to the top.
                    </p> */}

                    <div className="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 gap-16 p-16">
                        <CamapignCard />
                        <CamapignCard />
                        <CamapignCard />
                    </div>
                </section>
            </div>
        </>
    )
}

export default App;
