const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

export const deleteCampaign = async (campaignId: string) => {
    if (!confirm("Are you sure you want to delete this campaign? This will also remove its sessions.")) {
        return false;
    }

    try {
        const res = await fetch(`${baseUrl}/campaign/${campaignId}`, {
            method: "DELETE",
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.error || "Failed to delete campaign");
        }

        return true; // ✅ success
    } catch (err: unknown) {
        alert(err instanceof Error ? err.message : "Error deleting campaign");
        return false;
    }
};
