export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center gap-8 p-8">
      <div className="text-center">
        <h1 className="text-5xl font-bold mb-4 text-gray-800">
          Welcome to Dungeon Scribe
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Your AI-powered D&D session tracker and transcription tool
        </p>
      </div>

      <svg
        width="400px"
        height="250px"
        viewBox="0 0 512 512"
        xmlns="http://www.w3.org/2000/svg"
        className="text-gray-700">
        <path fill="currentColor" d="M248 20.3L72.33 132.6 248 128.8zm16 0v108.5l175.7 3.8zm51.4 58.9c6.1 3.5 8.2 7.2 15.1 4.2 10.7.8 22.3 5.8 27.6 15.7 4.7 4.5 1.5 12.6-5.2 12.6-9.7.1-19.7-6.1-14.6-8.3 4.7-2 14.7.9 10-5.5-3.6-4.5-11-7.8-16.3-5.9-1.6 6.8-9.4 4-12-.7-2.3-5.8-9.1-8.2-15-7.9-6.1 2.7 1.6 8.8 5.3 9.9 7.9 2.2.2 7.5-4.1 5.1-4.2-2.4-15-9.6-13.5-18.3 5.8-7.39 15.8-4.62 22.7-.9zm-108.5-3.5c5.5.5 12.3 3 10.2 9.9-4.3 7-9.8 13.1-18.1 14.8-6.5 3.4-14.9 4.4-21.6 1.9-3.7-2.3-13.5-9.3-14.9-3.4-2.1 14.8.7 13.1-11.1 17.8V92.3c9.9-3.9 21.1-4.5 30.3 1.3 8 4.2 19.4 1.5 24.2-5.7 1.4-6.5-8.1-4.6-12.2-3.4-2.7-8.2 7.9-7.5 13.2-8.8z" />
      </svg>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl">
        <div className="card bg-base-100 shadow-lg p-6 text-center">
          <h3 className="text-xl font-bold mb-2">Record Sessions</h3>
          <p className="text-gray-600">
            Record your D&D sessions with automatic speaker identification
          </p>
        </div>
        <div className="card bg-base-100 shadow-lg p-6 text-center">
          <h3 className="text-xl font-bold mb-2">AI Processing</h3>
          <p className="text-gray-600">
            Automatically extract characters, locations, and events
          </p>
        </div>
        <div className="card bg-base-100 shadow-lg p-6 text-center">
          <h3 className="text-xl font-bold mb-2">Track Everything</h3>
          <p className="text-gray-600">
            View timelines, character sheets, and campaign summaries
          </p>
        </div>
      </div>

      <div className="text-center mt-8">
        <p className="text-gray-500 text-sm">
          Use the navigation above to get started
        </p>
      </div>
    </div>
  );
}