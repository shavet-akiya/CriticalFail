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

                    <div className="carousel w-full py-8">
                        {/* Slide 1 */}
                        <div id="slide1" className="carousel-item relative w-full flex justify-center">
                            <CampaignBook />

                            <div className="absolute left-1 top-1/2 right-1 flex -translate-y-1/2 justify-between">
                                <a href="#slide2" className="btn btn-circle">❮</a>
                                <a href="#slide2" className="btn btn-circle">❯</a>
                            </div>
                        </div>

                        {/* Slide 2 */}
                        <div id="slide2" className="carousel-item relative w-full flex justify-center">
                            <CampaignBook />
                            <div className="absolute left-1 top-1/2 right-1 flex -translate-y-1/2 justify-between">
                                <a href="#slide1" className="btn btn-circle">❮</a>
                                <a href="#slide1" className="btn btn-circle">❯</a>
                            </div>
                        </div>
                    </div>

                    <div className="flex flex-row gap-4 justify-center items-center">
                        <button className="btn btn-neutral">Select campaign</button>
                        <button className="btn btn-outline obsidian-colour hover:bg-white">New campaign</button>

                    </div>
                </div>
            </div>
        </div>
    );
}


{/* <Toast type="error" message="Task failed successfully." />
<Toast message="New mail arrived." /> */}
