export type Graph = {
    graph_id: string,
    name: string
}

export type Conversation = {
    conversation_id: string,
    name: string,
    created_at: string
}

export type History = {
    user: string,
    assistant: string | null,
    timestamp: string
}