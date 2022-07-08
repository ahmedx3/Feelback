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
          <v-card class="mx-auto" max-width="344">
            <v-img src="../assets/Cinema-pattern.jpg" height="200px"></v-img>
            <v-card-title>
              <v-file-input
                show-size
                counter
                label="Add Trailer/Ad Video"
                v-model="trailerVideo"
                prepend-icon="mdi-video-plus"
              ></v-file-input>
            </v-card-title>
          </v-card>
        </v-col>
        <v-col cols="6">
          <v-card class="mx-auto" max-width="344">
            <v-img src="../assets/reaction.jpg" height="200px"></v-img>
            <v-card-title>
              <v-file-input
                show-size
                counter
                label="Add Reaction Video"
                v-model="reactionVideo"
                prepend-icon="mdi-video-plus"
              ></v-file-input>
            </v-card-title>
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
import api from '../api/index';

export default {
  name: 'Home',
  data() {
    return {
      frameRate: 0,
      trailerVideo: null,
      reactionVideo: null,
    };
  },
  methods: {
    startAnalytics() {
      const formdata1 = new FormData();
      formdata1.append('video', this.trailerVideo);
      formdata1.append('type', 'trailer');

      api.uploadVideo(formdata1).then((res) => {
        console.log(res.data.id);
        const trailerID = res.data.id;

        const formdata2 = new FormData();
        formdata2.append('video', this.reactionVideo);
        formdata2.append('type', 'reaction');
        formdata2.append('trailer_id', trailerID);

        api.uploadVideo(formdata2).then((res2) => {
          console.log(res2.data.id);
          const reactionID = res2.data.id;

          let rate = 0;

          if (this.frameRate === 0) {
            rate = 'native';
          } else {
            rate = this.frameRate;
          }

          const videoInfo = {
            fps: rate,
            save_annotated_video: true,
          };
          api.startProcessingVideo(reactionID, videoInfo).then(() => {
            // Route to loading page
            console.log(res);
            this.$router.push(`/dashboard/${reactionID}`);
          });
        });
      });
    },
  },
};
</script>
