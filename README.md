# CriticalFail
Dungeon Scribe - README
Version: 1.0 Release
Developers: Team Critical Fail

Application Overview:
Dungeon Scribe is an AI-powered tool designed to enhance the Dungeons & Dragons (D&D) tabletop role-playing experience. It captures real-time audio from game conversations and converts it to text using Whisper. It organises events, characters and their interactions, and locations for multiple D&D campaigns. The application emphasises data security and ethical AI usage, ensuring user data autonomy and privacy.

Features:

* Transcription of D&D sessions
* Session and in-game event tracking
* Character and location management
* Local embedding storage using ChromaDB

Prerequisites:

* Docker Desktop
* WSL enabled
* Internet connection for initial package installations

Installation & Setup:

1. Install Docker Desktop:

   * Download from [https://www.docker.com/products/docker-desktop/]
   * Follow instructions to enable WSL during installation.

2. Enable WSL (if not already enabled):

   * Open PowerShell as Administrator
   * Run: `wsl --install`
   * Restart your computer if prompted

3. Open the project root folder:

   * Navigate to the `CriticalFail` directory via the terminal
   * Set up an `.env` file with following:
  `NEXT_PUBLIC_API_BASE_URL=http://localhost:9000`

4. Build and run the application:

   * Run: `docker-compose up --build -d`
   * All dependencies will be automatically installed by Docker

5. Access the app:
   * Navigate to the `ui` directory via the terminal. 
   * Run `npm run build` and `npm run start` to get an optimised running of the application.
   * Open your browser and go to: [http://localhost:3000](http://localhost:3000)


Configuration Notes:

* Ensure that all local ports are available for hosting the application
* Local storage ensures that all session data remains on your device

Overview of required ports:

| Service | Description            | Port  |
| ------- | ---------------------- | ----- |
| server  | Backend API            | 9000  |
| ui      | Next.js frontend       | 3000  |
| speech  | Speech-to-text service | 8001  |
| chroma  | Vector database        | 8000  |
| ollama  | AI model service       | 11434 |

Volumes:

* `./server/src/recordings` → `/app/recordings`
* `./server/src/transcripts` → `/app/transcripts`
* `./server/images/campaign_images` → `/app/server/images/campaign_images`
* `./models` → `/app/models`
* `ollama` → `/root/.ollama` (persist AI models)

References:

* Dice SVG icon: [https://www.svgrepo.com/svg/322177/dice-twenty-faces-twenty](https://www.svgrepo.com/svg/322177/dice-twenty-faces-twenty)
* Services setup inspired by Docker and AI integrations.
* AI was used in the production of this codebase for debugging and refactoring.