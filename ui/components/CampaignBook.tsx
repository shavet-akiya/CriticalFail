export default function CampaignCard() {
    return (
        <div className="w-[22rem] h-[18rem] bg-[#f7f3ef] border-2 border-[#c8b39e] rounded-md shadow-md p-6 font-serif text-[#3c1642] leading-relaxed tracking-wide">
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
    );
}
