import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { NgModule } from '@angular/core';
import { AssistantComponent } from './pages/assistant/assistant.component';
import { GraphsComponent } from './pages/graphs/graphs.component';
import { GraphDetailsComponent } from './pages/graph-details/graph-details.component';

export const routes: Routes = [
    { path: 'home', component: HomeComponent }, 
    { path: 'assistant', component: AssistantComponent }, 
    { path: 'graphs', component: GraphsComponent },
    { path: 'graphs/details', component: GraphDetailsComponent },
    { path: 'graphs/details/:id', component: GraphDetailsComponent }, 
    { path: '**', redirectTo: 'home' } 
  ];

  @NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
  })
  export class AppRoutingModule { }