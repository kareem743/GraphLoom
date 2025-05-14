import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faPaperPlane, faPlus } from '@fortawesome/free-solid-svg-icons';
import { CommonService } from '../../shared/services/common.service';
import { AssistantService } from './assistant.service';
import { Graph, Conversation, History } from '../../shared/types/graph';
import { NgClass, NgFor } from '@angular/common';
import { NzSelectModule } from 'ng-zorro-antd/select';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzInputModule } from 'ng-zorro-antd/input'; 

@Component({
  selector: 'app-graph',
  imports: [
    MatInputModule,
    FormsModule,
    FontAwesomeModule,
    NgFor,
    NgClass,
    NzSelectModule,
    NzButtonModule,
    NzInputModule
  ], 
  templateUrl: './assistant.component.html',
  styleUrl: './assistant.component.css'
})
export class AssistantComponent implements OnInit {
  
  userId: string = '';
  graphs: Graph[] = [];
  conversations: Conversation[] = [];
  history: History[] = [];
  userMessage: string = '';
  
  selectedGraphId: string | null = null;
  selectedConversationId: string | null = null;
  
  sendIcon = faPaperPlane;
  plusIcon = faPlus;

  isTyping: boolean = false;

  constructor(private commonService: CommonService, private graphService: AssistantService) {}

  ngOnInit(): void {
    this.userId = this.commonService.getUserId()!;
    this.loadGraphs();
  }

  loadGraphs() {
    this.graphService.getUserGraphs(this.userId).subscribe(
      (data) => {
        this.graphs = data.graphs;
        console.log(this.graphs);
      },
      (error) => {
        console.error('Error:', error);
        this.graphs = [];
      }
    );
  }

  loadConversations() {
    this.graphService.getGraphConversations(this.userId, this.selectedGraphId!).subscribe(
      (data) => {
        this.conversations = data.conversations;
        if (this.conversations.length > 0) {
          this.selectedConversationId = this.conversations[0].conversation_id;
          this.loadHistory();
        } else {
          this.selectedConversationId = null;
          this.history = [];
        }
      },
      (error) => {
        console.error('Error:', error);
        this.conversations = [];
        this.selectedConversationId = null;
      }
    );
  }

  loadHistory() {
    this.graphService.getConversationHistory(this.selectedConversationId!).subscribe(
      (data) => {
        this.history = data.history;
      },
      (error) => {
        console.log('Error:', error);
        this.history = [];
      }
    );
  }

  postMessage(query: string) {
    this.isTyping = true;
    let requestObj: any = {
      conversation_id: this.selectedConversationId,
      query: query
    };

    this.graphService.sendMessage(requestObj).subscribe(
      (data) => {
        let response = data.response;
        this.history[this.history.length - 1].assistant = response;
        this.isTyping = false;
      },
      (error) => {
        console.log("Error:", error);
        this.isTyping = false;
      }
    );
  }

  postNewMessage(query: string) {
    this.isTyping = true;
    let requestObj: any = {
      user_id: this.userId,
      graph_id: this.selectedGraphId,
      query: query
    };

    this.graphService.sendNewMessage(requestObj).subscribe(
      (data) => {
        let response = data.response;
        this.history[this.history.length - 1].assistant = response;
        this.selectedConversationId = data.conversation_id;
        this.conversations.push({
          conversation_id: this.selectedConversationId!,
          name: data.conversation_name,
          created_at: new Date().toISOString()
        });
        this.isTyping = false;
      },
      (error) => {
        console.log("Error:", error);
        this.isTyping = false;
      }
    );
  }

  trackById(index: number, item: Graph) {
    return item.graph_id;
  }

  onDropdownSelectionChange(selectionId: string) {
    this.selectedGraphId = selectionId;
    if (this.selectedGraphId) {
      this.loadConversations();
    }
  }

  onConversationSelection(conversationId: string) {
    this.selectedConversationId = conversationId;
    this.loadHistory();
  }

  sendMessage() {
    if (!this.userMessage.trim()) {
      return;
    }
    this.history.push({
      user: this.userMessage.trim(),
      timestamp: new Date().toISOString(),
      assistant: null
    });
    if (this.selectedConversationId == null) {
      this.postNewMessage(this.userMessage.trim());
    } else {
      this.postMessage(this.userMessage.trim());
    }
    this.userMessage = '';
  }

  createNewConversation() {
    if (this.selectedGraphId) {
      this.loadConversations();
    }
    setTimeout(() => {
      this.selectedConversationId = null;
      this.history = [];
    }, 500);
  }

  formatText(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>')
      .replace(/```([\s\S]+?)```/g, '<pre><code>$1</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
      .replace(/\*(.*?)\*/g, '<i>$1</i>')
      .replace(/### (.*?)<br>/g, '<h3>$1</h3>')
      .replace(/## (.*?)<br>/g, '<h2>$1</h2>')
      .replace(/- (.*?)<br>/g, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>')
      .replace(/\d+\.\s(.*?)<br>/g, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)+/g, '<ol>$&</ol>')
      .replace(/\|(.+?)\|<br>/g, (match) => {
        const cells = match.split('|').slice(1, -1).map(cell => `<td>${cell.trim()}</td>`).join('');
        return `<tr>${cells}</tr>`;
      })
      .replace(/(<tr>.*<\/tr>)+/g, '<table>$&</table>')
      .replace(/:fire:/g, '🔥')
      .replace(/:rocket:/g, '🚀')
      .replace(/:smile:/g, '😃')
      .replace(/:check:/g, '✅');
  }
}