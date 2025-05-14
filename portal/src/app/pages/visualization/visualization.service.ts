import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

interface Graph {
  graph_id: string;
  name: string;
  folder_path: string;
  user_id: string;
  created_at: string;
}

interface GraphVisualization {
  nodes: { id: string; label: string; title: string }[];
  edges: { from: string; to: string; label: string; title: string }[];
}

@Injectable({
  providedIn: 'root'
})
export class VisualizationService {
    private apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  getGraphs(userId: string): Observable<{ graphs: Graph[] }> {
    return this.http.get<{ graphs: Graph[] }>(`${this.apiUrl}/graphs/?user_id=${userId}`);
  }

  getGraphVisualization(userId: string, graphId: string): Observable<GraphVisualization> {
    return this.http.get<GraphVisualization>(`${this.apiUrl}/graphs/${graphId}/visualization?user_id=${userId}`);
  }
}