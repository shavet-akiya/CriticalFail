import React, { useEffect, useState, createContext, useContext } from "react";

// Should match BackEnd\Event.py
interface Event {
    id: number;
    summary: string;
    characters: [string];
    places: [string];
    themes: [string];
    tags: [string];
}

interface Message {
  message: string;
}

const TodosContext = createContext({
  todos: [], fetchTodos: () => {}
})

export default function Todos() {
  const [todos, setTodos] = useState([])
  const fetchTodos = async () => {
    const response = await fetch("http://localhost:8000")
    const todos = await response.json()
    setTodos(todos.data)
  }
  useEffect(() => {
    fetchTodos()
  }, [])

  return (
    <>
    Hii
    {todos}
    </>
  )
}