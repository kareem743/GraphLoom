import { Component } from '@angular/core';
import { GraphsService } from './graphs.service';
import { Graph } from '../../shared/types/graph-types';
import { CommonService } from '../../shared/services/common.service';
import { faGear } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { NgClass } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { NzButtonModule } from 'ng-zorro-antd/button';

@Component({
  selector: 'app-graphs',
  imports: [FontAwesomeModule, NgClass, RouterModule, NzButtonModule],
  templateUrl: './graphs.component.html',
  styleUrl: './graphs.component.css'
})
export class GraphsComponent {
  userId: string = '';
  graphs: Graph[] = [];

  settingsIcon = faGear;

  constructor(private graphsService: GraphsService, private commonService: CommonService, private router: Router) {
    this.userId = this.commonService.getUserId()!;
  }

  ngOnInit() {
    this.loadGraphs();
  }

  loadGraphs() {
    this.graphsService.getUserGraphs(this.userId).subscribe(
      (data) => {
        this.graphs = data.graphs;
        this.graphs.forEach(graph => {
          const date = new Date(graph.created_at);
          const day = String(date.getDate()).padStart(2, '0');
          const month = String(date.getMonth() + 1).padStart(2, '0');
          const year = date.getFullYear();
          graph.created_at = `${day}/${month}/${year}`;
        });
      },
      (error) => {
        console.error('Error:', error);
        this.graphs = [];
      }
    );
  }

  createGraph() {
    this.router.navigate(['graphs', 'details']);
  }

  onSettingsClick(graph_id: string) {
    this.router.navigate(['graphs', 'details', graph_id]);
  }
}