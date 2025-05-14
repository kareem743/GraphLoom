export type Graph = {
    graph_id: string,
    name: string,
    folder_path: string,
    user_id: string,
    created_at: string,
    last_updated: string,
}

export type GraphDetails = {
    graph_id?: string | null,
    name?: string | null,
    folder_path?: string | null,
    user_id?: string | null,
    created_at?: string | null
}