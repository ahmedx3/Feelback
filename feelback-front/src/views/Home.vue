<template>
  <div class="home">
    <v-toolbar elevation="2" height="100">
      <v-spacer> </v-spacer>
      <v-img src="../assets/logo.png" max-height="125" contain> </v-img>
      <v-spacer> </v-spacer>
    </v-toolbar>
    <v-container fill-height class="my-10">
      <v-row justify="center" align="center">
        <v-col cols="6">
          <v-card class="mx-auto" max-width="344" @click="addTrailerVideo">
            <v-img src="../assets/Cinema-pattern.jpg" height="200px"></v-img>
            <v-card-title> Add Trailer Video </v-card-title>
          </v-card>
        </v-col>
        <v-col cols="6">
          <v-card class="mx-auto" max-width="344" @click="uploadReactionVideo">
            <v-img src="../assets/reaction.jpg" height="200px"></v-img>
            <v-card-title> Add Reaction Video </v-card-title>
          </v-card>
        </v-col>
      </v-row>
      <v-row justify="center" align="center" class="my-5">
        <h1><v-icon large>mdi-cog</v-icon> Configurations</h1>
      </v-row>
      <v-row justify="center" align="center" class="my-1">
        <v-form ref="form">
          <v-text-field
            v-model="frameRate"
            type="number"
            label="Frame Rate"
            required
          ></v-text-field>

          <v-btn @click="startAnalytics"
            >Start Analytics
            <v-icon right dark> mdi-chart-areaspline </v-icon>
          </v-btn>
        </v-form>
      </v-row>
    </v-container>
  </div>
</template>

<script>
// @ is an alias to /src
// import HelloWorld from '@/components/HelloWorld.vue';

export default {
  name: 'Home',
  methods: {
    addTrailerVideo() {
      // TODO : Add functionality here
    },
    uploadReactionVideo() {
      // TODO : Add functionality here
    },
    startAnalytics() {
      // TODO : Add functionality here
    },
  },
};
</script>
