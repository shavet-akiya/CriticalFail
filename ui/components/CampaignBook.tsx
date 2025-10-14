export default function CampaignBook() {
    return (
        <div className="relative w-[22rem] h-[18rem] bg-[#f7f3ef] border-[2px] border-[#c8b39e] rounded-md shadow-2xl p-6 font-serif text-[#3c1642] leading-relaxed tracking-wide overflow-hidden">
            {/* Spine */}
            <div className="absolute left-0 top-0 h-full w-5 bg-gradient-to-r from-[#3c1642] via-[#4a2455] to-[#5e3a6b] rounded-l-md shadow-inner"></div>

            {/* Subtle overlay for paper texture */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-[#e9ddd2]/50 rounded-md pointer-events-none"></div>

            {/* Book content (ensure it's positioned inside padding area) */}
            <div className="relative z-10 pl-6 pr-4">
                <h2 className="text-2xl font-bold mb-3 text-center italic">
                    The Wild Beyond the Witchlight
                </h2>

                <div className="border-t border-[#d9cfc5] pt-3">
                    <p className="indent-5 mb-2">
                        Bound in tales and twilight, this tome holds the whispered secrets of the Feywild.
                    </p>
                    <p className="indent-5">
                        A story of laughter and loss, of carnival lights that never fade, and of doors best left unopened.
                    </p>
                </div>
            </div>

            {/* Page edge (right side) */}
            <div className="absolute right-0 top-0 h-full w-[6px] bg-gradient-to-l from-[#e3d5c9] to-[#f7f3ef] rounded-r-md"></div>
        </div>
    );
}
