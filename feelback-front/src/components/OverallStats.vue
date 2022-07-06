<template>
  <div>
    <h1 class="title blue--text">GENERAL STATISTICS</h1>

    <v-row class="mt-3">
      <v-col cols="4" class="stat">
        <h3 class="blue--text text-h2 mb-2">
          {{ stats && stats.number_of_persons ? Math.floor(stats.number_of_persons / 10) : 0
          }}{{ stats && stats.number_of_persons ? stats.number_of_persons % 10 : 0 }}
        </h3>
        <p class="text-overline">Viewers</p>
      </v-col>
      <v-col cols="4" class="stat">
        <h3 class="blue--text text-h2 mb-2">00</h3>
        <p class="text-overline">Key Moments</p>
      </v-col>
      <v-col cols="4" class="stat">
        <h3 class="blue--text text-h2 mb-2">{{ topEmotion }}</h3>
        <p class="text-overline">Top Emotion</p>
      </v-col>
    </v-row>

    <!-- Graphs -->
    <v-row>
      <v-col cols="6">
        <PieChart
          title="Emotions"
          :data="stats ? stats.emotions : null"
          :colors="{
            Happy: '#FFEB3B',
            Sadness: '#009688',
            Disgust: '#673AB7',
            Neutral: '#8BC34A',
            Surprise: '#0091EA',
          }"
        />
      </v-col>
      <v-col cols="6">
        <PieChart
          title="Focus"
          :data="
            stats
              ? {
                  Yes: stats.attention * 100,
                  No: 100 - stats.attention * 100,
                }
              : null
          "
          :colors="{
            Yes: '#2196F3',
            No: '#F44336',
          }"
        />
      </v-col>
      <v-col cols="6">
        <PieChart
          title="Gender"
          :data="stats ? stats.gender : null"
          :colors="{
            Male: '#2196F3',
            Female: '#F44336',
          }"
        />
      </v-col>
      <v-col cols="6">
        <PieChart
          title="Age"
          :data="stats ? ViewerAge : null"
          :colors="{
            children: '#9C27B0',
            youth: '#E91E63',
            adults: '#3F51B5',
            seniors: '#FF9800',
          }"
        />
      </v-col>
    </v-row>
  </div>
</template>

<script>
import PieChart from './charts/PieChart.vue';

export default {
  components: {
    PieChart,
  },

  props: {
    stats: Object,
  },

  computed: {
    topEmotion() {
      if (this.stats && this.stats.emotions) {
        let topEmotion;
        let maxPercentage = 0;
        Object.entries(this.stats.emotions).forEach((object) => {
          const [key, value] = object;
          if (value > maxPercentage) {
            topEmotion = key;
            maxPercentage = value;
          }
        });

        return topEmotion;
      }
      return 'None';
    },

    ViewerAge() {
      const value = {
        children: 0,
        youth: 0,
        adults: 0,
        seniors: 0,
      };
      if (this.stats && this.stats.age) {
        this.stats.age.forEach((entry) => {
          if (entry <= 14) {
            value.children += 1;
          } else if (entry <= 22) {
            value.youth += 1;
          } else if (entry <= 40) {
            value.adults += 1;
          } else {
            value.seniors += 1;
          }
        });
      }
      return value;
    },
  },
};
</script>

<style>
.title {
  text-align: center;
  margin-top: 10px;
}

.stat {
  text-align: center;
}
</style>
