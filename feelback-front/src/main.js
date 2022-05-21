import 'zingchart/es6';
import Vue from 'vue';
import zingchartVue from 'zingchart-vue';
import App from './App.vue';
import store from './store';
import router from './router';
import vuetify from './plugins/vuetify';
import './registerServiceWorker';

Vue.config.productionTip = false;
Vue.component('zingchart', zingchartVue);

new Vue({
  router,
  store,
  vuetify,
  render: (h) => h(App),
}).$mount('#app');
