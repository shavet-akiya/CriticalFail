export default function newSession() {
  return (
    <div className="flex flex-col items-center">
      <div>
        <img
          src="/svg/recording_page_dice.svg"
          alt="D20"
          className="w-1/2 h-auto" />
      </div>

      <div className="flex items-center h-screen justify-center">
        <div className="flex">
          <button className="btn btn-outline">Upload Session</button>
          <button className="btn btn-outline">Start Recording</button>
        </div>
      </div>
    </div>
  );
}
