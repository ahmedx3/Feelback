<template>
  <div class="home">
    <v-toolbar elevation="2" height="100">
      <v-spacer> </v-spacer>
      <v-img src="../assets/logo.png" max-height="125" contain> </v-img>
      <v-spacer> </v-spacer>
    </v-toolbar>
    <v-container fill-height class="my-10">
      <v-row justify="center" align="center">
        <h1>Demos</h1>
      </v-row>
      <v-row>
        <v-card class="mx-auto my-5" max-width="344" v-for="x in 6" :key="x">
          <v-img src="../assets/Thor_Moment.jpg" height="200px"></v-img>
          <v-card-title> Demo 1 </v-card-title>
        </v-card>
      </v-row>
    </v-container>
  </div>
</template>

<script>
// @ is an alias to /src
// import HelloWorld from '@/components/HelloWorld.vue';

export default {
  name: 'Demos',
};
</script>
