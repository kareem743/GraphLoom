import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AssistantService {

  private apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  getUserGraphs(userId: string): Observable<any> {
    let params = new HttpParams().set('user_id', userId);
    return this.http.get(`${this.apiUrl}/graphs`, { params });
  }

  getGraphConversations(userId: string, graphId: string): Observable<any> {
    let params = new HttpParams()
      .set('user_id', userId)
      .set('graph_id', graphId);
    return this.http.get(`${this.apiUrl}/conversations`, { params });                   
  }

  getConversationHistory(conversationId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/conversations/${conversationId}/history`);                   
  }

  sendMessage(request: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/conversations/send-message`, request);
  }

  sendNewMessage(request: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/conversations/start-chat`, request);
  }
}