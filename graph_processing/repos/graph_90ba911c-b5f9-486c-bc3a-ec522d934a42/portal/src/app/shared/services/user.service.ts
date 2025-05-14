import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private apiUrl = 'http://127.0.0.1:8000/users';

  constructor(private http: HttpClient) {}

  // Function to get user info by ID
  getUserById(userId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/${userId}`)
  }
}
