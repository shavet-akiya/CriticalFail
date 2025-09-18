// /app/timeline/page.tsx
'use client';

import React from 'react';
import { EventsProvider, useEvents } from './events';

function Timeline() {
  const { events } = useEvents();

  return (
    <div className="max-w-screen-xl pt-24">
      <h1 className="text-2xl font-bold mb-4">Timeline</h1>
      <ul className="space-y-4">
        {events.map((event) => (
          <li key={event.id} className="border-l-4 border-blue-500 pl-4">
            <p className="text-lg font-semibold">{event.summary}</p>
            <p className="text-sm text-gray-600">
              Characters: {event.characters.join(", ")}
            </p>
            <p className="text-sm text-gray-600">
              Places: {event.places.join(", ")}
            </p>
            <p className="text-sm text-gray-600">
              Themes: {event.themes.join(", ")}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Page() {
  return (
    <EventsProvider>
      <Timeline />
    </EventsProvider>
  );
}
