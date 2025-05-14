import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class GraphsService {

  private apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  getUserGraphs(userId: string): Observable<any> {
    let params = new HttpParams().set('user_id', userId);
    return this.http.get(`${this.apiUrl}/graphs`, { params });   
  }

  getGraphDetails(userId: string, graph_id: string): Observable<any> {
    let params = new HttpParams().set('user_id', userId);
    return this.http.get(`${this.apiUrl}/graphs/${graph_id}`, { params });   
  }

  createGraph(request: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/graphs/create`, request);
  }

  deleteGraph(graph_id: string, user_id: string): Observable<any> {
    let params = new HttpParams().set('user_id', user_id);
    return this.http.delete(`${this.apiUrl}/graphs/${graph_id}`, { params });   
  }
}