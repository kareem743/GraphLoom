import { Component, OnInit, AfterViewInit, ElementRef, ViewChild } from '@angular/core';
import { VisualizationService } from './visualization.service';
import { ActivatedRoute } from '@angular/router';
import { Network, Data } from 'vis-network/standalone'; // Use standalone to avoid peer dependencies
import { DataSet } from 'vis-data'; // Import DataSet from vis-data
import { FormsModule } from '@angular/forms';
import { NgFor } from '@angular/common';
import { NzOptionComponent, NzSelectModule } from 'ng-zorro-antd/select';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzInputModule } from 'ng-zorro-antd/input';

@Component({
  selector: 'app-graph-visualization',
  templateUrl: './visualization.component.html',
  styleUrls: ['./visualization.component.css'],
  standalone: true,
  imports: [FormsModule, NgFor,     NzSelectModule,
      NzButtonModule,
      NzInputModule]
})
export class GraphVisualizationComponent implements OnInit, AfterViewInit {
  @ViewChild('network') networkContainer!: ElementRef;
  userId: string = '123'; // TODO: Replace with auth service
  graphs: any[] = [];
  selectedGraphId: string | null = null;
  network: Network | null = null;
  nodesDataSet: DataSet<any> | null = null; // Use any for flexibility, or define Node interface

  constructor(
    private graphService: VisualizationService,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    this.loadGraphs();
    this.route.queryParams.subscribe(params => {
      this.userId = params['userId'] || '123';
    });
  }

  ngAfterViewInit(): void {
    this.initializeNetwork();
  }
  trackById(index: number, item: any) {
    return item.graph_id;
  }
  loadGraphs(): void {
    this.graphService.getGraphs(this.userId).subscribe({
      next: (response) => {
        this.graphs = response.graphs;
        if (this.graphs.length > 0) {
          this.selectedGraphId = this.graphs[0].graph_id;
          this.loadGraphVisualization();
        }
      },
      error: (err) => console.error('Failed to load graphs:', err)
    });
  }

  onGraphSelect(): void {
    if (this.selectedGraphId) {
      this.loadGraphVisualization();
    }
  }

  loadGraphVisualization(): void {
    if (!this.selectedGraphId || !this.network) return;

    this.graphService.getGraphVisualization(this.userId, this.selectedGraphId).subscribe({
      next: (data) => {
        this.nodesDataSet = new DataSet(data.nodes.map(node => ({
          id: node.id,
          label: node.label,
          title: node.title
        })));

        const edges = new DataSet(data.edges.map((edge, index) => ({
          id: `edge_${index}`,
          from: edge.from,
          to: edge.to,
          label: edge.label,
          title: edge.title
        })));

        const networkData: Data = { nodes: this.nodesDataSet, edges };
        this.network!.setData(networkData);
        this.network!.setOptions({
          nodes: {
            shape: 'dot',
            size: 20,
            font: { size: 12 },
            borderWidth: 2
          },
          edges: {
            width: 2,
            arrows: { to: { enabled: true, scaleFactor: 1 } },
            font: { size: 10, align: 'middle' }
          },
          physics: {
            forceAtlas2Based: {
              gravitationalConstant: -50,
              centralGravity: 0.01,
              springLength: 100
            },
            minVelocity: 0.75
          },
          interaction: {
            hover: true,
            zoomView: true,
            dragView: true
          }
        });
      },
      error: (err) => console.error('Failed to load graph visualization:', err)
    });
  }

  initializeNetwork(): void {
    const container = this.networkContainer.nativeElement;
    this.nodesDataSet = new DataSet([]);
    const edges = new DataSet([]);
    const data: Data = { nodes: this.nodesDataSet, edges };
    this.network = new Network(container, data, {});

    // Attach click event listener
    this.network.on('click', (params) => {
      if (params.nodes.length > 0 && this.nodesDataSet) {
        const nodeId = params.nodes[0];
        const node = this.nodesDataSet.get(nodeId);
        if (node) {
          console.log('Node clicked:', node);
          // TODO: Show node details in a sidebar/modal
        }
      }
    });
  }
}