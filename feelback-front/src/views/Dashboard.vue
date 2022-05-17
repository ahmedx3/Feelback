<template>
  <v-container>
    <v-row>
      <v-col cols="6">
        <p class="font-weight-medium text-h4 center-text">Trailer</p>
        <v-col cols="12">
          <video controls width="550" height="275" autoplay="true" muted="false">
            <source src="../assets/Thor_Trailer.mp4" type="video/webm" />
            Sorry, your browser doesn't support embedded videos.
          </video>
        </v-col>
      </v-col>
      <v-col cols="6">
        <p class="font-weight-medium text-h4 center-text">Reaction</p>
        <v-col cols="12">
          <video controls width="550" height="275" autoplay="true" muted="false">
            <source src="../assets/Thor_Reaction.mp4" type="video/webm" />
            Sorry, your browser doesn't support embedded videos.
          </video>
        </v-col>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  data() {
    return {};
  },
};
</script>

<style>
.center-text {
  justify-content: center;
  text-align: center;
}
</style>
