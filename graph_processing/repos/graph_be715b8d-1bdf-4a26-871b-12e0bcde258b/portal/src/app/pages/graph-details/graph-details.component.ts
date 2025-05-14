import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ActionMode } from '../../shared/types/common';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzFormModule } from 'ng-zorro-antd/form'; 
import { NzInputModule } from 'ng-zorro-antd/input'; 
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { GraphDetails } from '../../shared/types/graph-types';
import { GraphsService } from '../graphs/graphs.service';
import { CommonService } from '../../shared/services/common.service';
import { NgIf } from '@angular/common';
import { NzMessageService } from 'ng-zorro-antd/message';
import { NzSpinModule } from 'ng-zorro-antd/spin';

@Component({
  selector: 'app-graph-details',
  imports: [
    NzButtonModule,
    ReactiveFormsModule,
    NzFormModule,
    NzInputModule,
    FormsModule,
    NgIf,
    NzSpinModule
  ],
  templateUrl: './graph-details.component.html',
  styleUrl: './graph-details.component.css'
})
export class GraphDetailsComponent implements OnInit {
  userId: string = '';
  graphId: string | null = null;
  graphDataSource: GraphDetails = {
    graph_id: null,
    name: null,
    folder_path: null,
    user_id: null,
    created_at: null
  };

  action: ActionMode = ActionMode.VIEW;
  addAction: ActionMode = ActionMode.ADD;
  viewAction: ActionMode = ActionMode.VIEW;

  isLoading = false;

  pageTitle: string = "Create Graph";
  graphForm: FormGroup;

  constructor(
    private route: ActivatedRoute,
    private graphsService: GraphsService,
    private commonService: CommonService,
    private fb: FormBuilder,
    private message: NzMessageService,
    private router: Router
  ) {
    this.userId = this.commonService.getUserId()!;
    this.graphForm = this.fb.group({
      name: ['', Validators.required],
      folder_path: ['', Validators.required],
      created_at: ['']
    });

    this.graphForm.valueChanges.subscribe(values => {
      this.graphDataSource = {
        ...this.graphDataSource,
        ...values
      };
    });
  }

  ngOnInit() {
    this.graphId = this.route.snapshot.paramMap.get('id');
    if (this.graphId == null) {
      this.initializeAddMode();
    } else {
      this.initializeViewMode();
    }
  }

  initializeAddMode() {
    this.action = ActionMode.ADD;
    this.pageTitle = "Create Graph";
    this.graphDataSource = {
      graph_id: null,
      name: null,
      folder_path: null,
      user_id: null,
      created_at: null
    };
    this.graphForm.reset(this.graphDataSource);
    this.updateFormState();
  }

  initializeViewMode() {
    this.action = ActionMode.VIEW;
    this.pageTitle = "Graph Details";
    this.updateFormState();
    this.loadGraphDetails();
  }

  loadGraphDetails() {
    this.isLoading = true;
    this.graphsService.getGraphDetails(this.userId, this.graphId!).subscribe(
      (data) => {
        this.graphDataSource = data;
        const date = new Date(this.graphDataSource.created_at!);
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const year = date.getFullYear();
        this.graphDataSource.created_at = `${day}/${month}/${year}`;
        this.graphForm.patchValue(this.graphDataSource);
        this.isLoading = false;
      },
      (error) => {
        console.error('Error:', error);
        this.graphDataSource = {
          graph_id: null,
          name: null,
          folder_path: null,
          user_id: null,
          created_at: null
        };
        this.graphForm.patchValue(this.graphDataSource);
        this.updateFormState();
        this.isLoading = false;
      }
    );
  }

  updateFormState() {
    if (this.action === ActionMode.ADD) {
      this.graphForm.get('name')?.enable();
      this.graphForm.get('folder_path')?.enable();
      this.graphForm.get('created_at')?.disable();
    } else if (this.action === ActionMode.VIEW) {
      this.graphForm.disable();
    }
  }

  onCreate() {
    if (this.graphForm.valid) {
      this.isLoading = true;
      let requestObj = {
        user_id: this.userId,
        graph_name: this.graphDataSource.name,
        folder_path: this.graphDataSource.folder_path
      };
      this.graphsService.createGraph(requestObj).subscribe(
        (response) => {
          this.isLoading = false;
          this.graphId = response.graph_id;
          this.message.create('success', response.message);
          this.initializeViewMode();
          this.loadGraphDetails();
        },
        (error) => {
          this.isLoading = false;
          this.message.create('error', error.error.detail || 'Failed to create graph');
        }
      );
    }
  }

  onDelete() {
    this.graphsService.deleteGraph(this.graphId!, this.userId).subscribe(
      (response) => {
        this.message.create('success', response.message);
        this.router.navigate(['graphs']);
      },
      (error) => {
        this.message.create('error', error.error.detail || 'Failed to delete graph');
      }
    );
  }
} 