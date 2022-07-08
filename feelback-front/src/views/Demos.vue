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
        <v-card
          class="mx-auto my-5"
          max-width="344"
          v-for="video in allVideos"
          :key="video"
          @click="goToDashboard(video.reaction_id)"
        >
          <v-img :src="video.img_url" height="200px"></v-img>
          <v-card-title> {{ video.filename }} </v-card-title>
          <v-card-subtitle>
            <h6 class="subtitle-2">duration: {{ video.duration }} seconds</h6>
          </v-card-subtitle>
        </v-card>
      </v-row>
    </v-container>
  </div>
</template>

<script>
import api from '../api/index';

export default {
  name: 'Demos',
  data() {
    return {
      allVideos: [],
    };
  },
  methods: {
    getAllReactionVideosInformation() {
      api.getAllReactionVideosInformation().then((res) => {
        this.allVideos = res.data;
      });
    },
    goToDashboard(reactionId) {
      this.$router.push({
        name: 'Dashboard',
        params: {
          id: reactionId,
        },
      });
    },
  },
  mounted() {
    this.getAllReactionVideosInformation();
  },
};
</script>
