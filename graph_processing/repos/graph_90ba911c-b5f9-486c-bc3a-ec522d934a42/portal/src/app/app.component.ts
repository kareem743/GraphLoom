import { Component, OnInit } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatButtonModule } from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon'
import { NgFor } from '@angular/common';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faHome, faMessage, faEye, faNetworkWired, faGear, faAngleDown } from '@fortawesome/free-solid-svg-icons';
import { icon } from '@fortawesome/fontawesome-svg-core';
import { User } from './shared/types/user';
import { UserService } from './shared/services/user.service';
import { HttpClientModule } from '@angular/common/http';
import { CommonService } from './shared/services/common.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, 
            MatSidenavModule, 
            NgFor, 
            MatButtonModule, 
            RouterLink, 
            RouterLinkActive,
            MatIconModule,
            FontAwesomeModule,
            HttpClientModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  title = 'RAGNAR';
  downArrow = faAngleDown;
  navItems = [
    { label: 'Home', icon: faHome, route: '/home' },
    { label: 'Assistant', icon: faMessage, route: '/assistant' },
    { label: 'Visualization', icon: faEye, route: '/visualization' },
    { label: 'Graphs', icon: faNetworkWired, route: '/graphs' },
    { label: 'Configuration', icon: faGear, route: '/configuration' },
  ];

    isNavOpen = true;

    userId: string = ''
    userData: User = {
      name: '',
      role: '',
      org: ''
    }

    constructor(private userService: UserService, private commonService: CommonService) {
      this.userId = '123'
      this.commonService.setUserId(this.userId);
    }

    ngOnInit(): void {
        this.getUserInfo();
    }

    getUserInfo() {
      this.userData = {
        name: "Hamdi",
        role: "Admin",
        org: "GraphLoom"
      }
    }

    toggleNav(): void {
      this.isNavOpen = !this.isNavOpen;
    }
  
}
