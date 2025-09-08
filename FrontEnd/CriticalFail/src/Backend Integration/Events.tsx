import React, { useEffect, useState, createContext, useContext } from "react";
import {
  Box,
  Button,
  Container,
  Flex,
  Input,
  DialogBody,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogRoot,
  DialogTitle,
  DialogTrigger,
  Stack,
  Text,
  DialogActionTrigger,
} from "@chakra-ui/react";

// Should match BackEnd\Event.py
interface Event {
    id: number;
    summary: string;
    characters: [string];
    places: [string];
    themes: [string];
    tags: [string];
}

const EventsContext = createContext({
  events: [], fetchEvents: () => {}
})

export default function Events() {
  const [events, setEvents] = useState([])
  const fetchEvents = async () => {
    const response = await fetch("http://localhost:8000/event")
    const events = await response.json()
    setEvents(events.data)
  }
  
  useEffect(() => {fetchEvents()}, [])

  return (
    <EventsContext.Provider value={{events, fetchEvents}}>
      <Container maxW="container.xl" pt="100px">
        Here begins the events
        <Stack gap={5}>
          {events.map((event: Event) => (
            <>
            Look, its event {event.id}!!!
              <dl>
                <dt><b>Summary</b></dt><dd>- {event.summary}</dd>
                <dt><b>Characters</b></dt><dd>- {event.characters.join(' and ')}</dd>
                <dt><b>Places</b></dt><dd>- {event.places.join(', ')}</dd>
                <dt><b>Themes</b></dt><dd>- {event.themes.join(', ')}</dd>
              </dl>
            </>
          ))}
        </Stack>
      </Container>
    </EventsContext.Provider>
  )
}