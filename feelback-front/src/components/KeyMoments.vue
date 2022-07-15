<template>
  <v-container>
    <h1 class="title blue--text">KEY MOMENTS</h1>

    <!-- Key Moments -->

    <v-row dense class="mt-3 mb-7">
      <v-col cols="4" v-for="(moment, i) in KeyMoments" :key="i">
        <v-card class="mx-auto" max-width="344">
          <v-carousel cycle height="200" hide-delimiter-background show-arrows-on-hover>
            <v-carousel-item>
              <v-img :src="moment.img_url" height="200px"></v-img>
            </v-carousel-item>
            <v-carousel-item>
              <v-img :src="moment.trailer_img_url" height="200px"></v-img>
            </v-carousel-item>
          </v-carousel>

          <v-card-title style="justify-content: center"> Key Moment {{ i + 1 }} </v-card-title>

          <div class="key-stats text-overline grey--text">
            <div>Start : {{ Math.floor(moment.start) }}</div>
            <div>End : {{ Math.floor(moment.end) }}</div>
          </div>

          <v-card-actions style="justify-content: center" class="pt-0 mt-0">
            <v-btn
              color="blue lighten-2"
              text
              @click="seekToCertainSecond(Math.floor(moment.start))"
            >
              Seek To Start</v-btn
            >
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  props: {
    KeyMoments: Array,
    seekToCertainSecond: Function,
  },
};
</script>

<style>
.title {
  text-align: center;
  margin-top: 10px;
}

.key-stats {
  display: flex;
  flex-direction: row;
  justify-content: space-around;
}
</style>
