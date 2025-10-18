"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Toast from "@/components/Toast";

export default function NewCampaign() {
    const router = useRouter(); // <-- get router
    const [campaignName, setCampaignName] = useState("");
    const [image, setImage] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [toast, setToast] = useState<{ type: "success" | "error" | "info"; message: string } | null>(null);

    useEffect(() => {
        if (toast) {
            const timer = setTimeout(() => setToast(null), 3000);
            return () => clearTimeout(timer);
        }
    }, [toast]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (!campaignName.trim()) {
            setToast({ type: "error", message: "Campaign name is required." });
            return;
        }

        // Show success toast
        setToast({ type: "success", message: `Campaign "${campaignName}" created successfully!` });

        // Optional: delay navigation to allow toast to appear briefly
        setTimeout(() => {
            router.push("/");
        }, 500); // delay to allow the toast to show

        // Reset form
        setCampaignName("");
        setImage(null);
        setPreview(null);
    };

    return (
        <div className="relative">
            {toast && <Toast type={toast.type} message={toast.message} />}

            <form onSubmit={handleSubmit} className="flex flex-col gap-8 bg-[#eff1ed] p-8 max-w-lg mx-auto">
                <h1 className="obsidian-colour text-4xl pb-4 text-center">
                    Create a Campaign
                </h1>

                <fieldset className="fieldset">
                    <legend className="fieldset-legend obsidian-colour font-semibold">Campaign Name</legend>
                    <input
                        type="text"
                        className="input input-neutral w-full"
                        placeholder="e.g. The Wild Beyond the Witchlight"
                        value={campaignName}
                        onChange={(e) => setCampaignName(e.target.value)}
                        required
                    />
                    <p className="label text-sm text-gray-500">Required</p>
                </fieldset>

                <fieldset className="fieldset">
                    <legend className="fieldset-legend font-semibold">Campaign Image</legend>
                    <input type="file" accept="image/*" className="file-input w-full" onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (!file) return;
                        setImage(file);
                        setPreview(URL.createObjectURL(file));
                    }} />
                    <label className="label text-sm text-gray-500">Max size 2MB</label>
                </fieldset>

                {preview && <img src={preview} alt="Campaign preview" className="rounded-xl shadow-lg max-h-64 object-contain border border-gray-300 mx-auto" />}

                <button type="submit" className="btn btn-primary w-full">Create</button>
            </form>
        </div>
    );
}
