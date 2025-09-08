import { ChakraProvider } from '@chakra-ui/react'
import { defaultSystem } from "@chakra-ui/react"
import Todos from "./Backend Integration/Events.tsx"

function App() {

  return (
    <ChakraProvider value={defaultSystem}>
      <Todos />
    </ChakraProvider>
  )
}

export default App;