<template>
  <div v-if="InDepthStats">
    <h1 class="title blue--text">In-Depth Analytics</h1>

    <v-row justify="center">
      <v-col cols="4">
        <v-select
          :items="['Horizontal', 'Aggregate', 'Mood']"
          label="Graph Type"
          v-model="SelectedChartType"
          dense
          filled
          @change="graphChanged"
        ></v-select>
      </v-col>
      <v-col cols="4">
        <v-select
          :items="criterias"
          label="Selected Criteria"
          v-model="criteriaType"
          dense
          filled
          :disabled="SelectedChartType === 'Mood'"
        ></v-select>
      </v-col>
    </v-row>
    <v-row justify="center">
      <v-col cols="12" class="text-center">
        <h4 class="blue--text">Filters</h4>
      </v-col>
      <v-col cols="3">
        <v-select
          :items="['All', 'Male', 'Female']"
          label="Gender"
          v-model="genderFilter"
          dense
          :disabled="SelectedChartType === 'Mood'"
        ></v-select>
      </v-col>
      <v-col cols="3">
        <v-select
          :items="['All', 'Children', 'Youth', 'Adults', 'Seniors']"
          label="Age"
          v-model="ageFilter"
          dense
          :disabled="SelectedChartType === 'Mood'"
        ></v-select>
      </v-col>
    </v-row>

    <!-- Charts -->

    <transition name="fade">
      <!-- Horizontal Charts -->
      <div v-if="SelectedChartType === 'Horizontal' && criteriaType === 'Emotions'">
        <HorizontalChart
          :data="cleanData(InDepthStats, 'Emotions', 'Horizontal')"
          :scaleY="{
            label: {
              text: 'Emotions',
              fontSize: 16,
            },
            values: ['Disguist 🤢', 'Sad 😞', 'Neutral 😐', 'Happy 😊', 'Surprise 😲'],
          }"
        />
      </div>

      <HorizontalChart
        v-if="SelectedChartType === 'Horizontal' && criteriaType === 'Attention'"
        :data="cleanData(InDepthStats, 'Attention', 'Horizontal')"
        :scaleY="{
          // set scale label
          label: {
            text: 'Attention',
            fontSize: 16,
          },
          values: ['Distracted', 'Focused'],
        }"
      />

      <!-- Area Charts -->
      <AreaChart
        v-if="SelectedChartType === 'Aggregate'"
        :data="cleanData(InDepthStats, criteriaType, 'Aggregate')"
      />

      <!-- Mood Chart -->
      <div>
        <LineChart v-if="SelectedChartType === 'Mood'" :data="getMoodData(moodData)" />
      </div>
    </transition>
  </div>
</template>

<script>
import HorizontalChart from './charts/HorizontalChart.vue';
import AreaChart from './charts/AreaChart.vue';
import LineChart from './charts/LineChart.vue';

export default {
  components: {
    HorizontalChart,
    AreaChart,
    LineChart,
  },

  props: {
    InDepthStats: Array,
    moodData: Array,
  },

  data() {
    return {
      // Graph controls
      SelectedChartType: 'Horizontal',
      criteriaType: 'Emotions',

      // Filters
      ageFilter: 'All',
      genderFilter: 'All',

      // Criterias
      criterias: ['Emotions', 'Attention'],
    };
  },

  methods: {
    getMoodData(moods) {
      const values = [];
      console.log(moods);
      moods.forEach((mood) => {
        values.push(mood.mood);
      });

      return [
        {
          aspect: 'spline',
          values,
        },
      ];
    },

    cleanData(persons, criteria, type) {
      const data = [];

      persons.forEach((person, index) => {
        const values = [];

        // Check the filters
        if (this.ageFilter === 'Children' && person.age > 14) return;
        if (this.ageFilter === 'Youth' && (person.age > 20 || person.age <= 14)) return;
        if (this.ageFilter === 'Adults' && (person.age > 40 || person.age <= 20)) return;
        if (this.ageFilter === 'Seniors' && person.age < 40) return;
        if (this.genderFilter === 'Male' && person.gender === 'Female') return;
        if (this.genderFilter === 'Female' && person.gender === 'Male') return;

        // Add the data
        if (criteria === 'Emotions' && type === 'Horizontal') {
          person.emotions.forEach((emotion) => {
            values.push(this.mapEmotions(emotion.emotion));
          });
        } else if (criteria === 'Attention' && type === 'Horizontal') {
          person.attention.forEach((attention) => {
            values.push(attention.attention);
          });
        } else if (criteria === 'Attention' && type === 'Aggregate') {
          let sum = 0;
          person.attention.forEach((attention) => {
            sum += attention.attention;
            values.push(sum);
          });
        } else if (criteria === 'Happiness' && type === 'Aggregate') {
          let sum = 0;
          person.emotions.forEach((emotion) => {
            sum += emotion.emotion === 'Happy' ? 1 : 0;
            values.push(sum);
          });
        } else if (criteria === 'Sadness' && type === 'Aggregate') {
          let sum = 0;
          person.emotions.forEach((emotion) => {
            sum += emotion.emotion === 'Sad' ? 1 : 0;
            values.push(sum);
          });
        } else if (criteria === 'Disgust' && type === 'Aggregate') {
          let sum = 0;
          person.emotions.forEach((emotion) => {
            sum += emotion.emotion === 'Disgust' ? 1 : 0;
            values.push(sum);
          });
        } else if (criteria === 'Surprisement' && type === 'Aggregate') {
          let sum = 0;
          person.emotions.forEach((emotion) => {
            sum += emotion.emotion === 'Surprise' ? 1 : 0;
            values.push(sum);
          });
        }

        if (index < 4) {
          const colors = ['#D31E1E', '#29A2CC', '#7CA82B', '#FF00FF'];
          data.push({
            aspect: 'spline',
            values,
            text: `Person ${person.id}`,
            'line-color': colors[person.id],
            'background-color': colors[person.id],
            marker: {
              'background-color': colors[person.id],
              'border-color': colors[person.id],
            },
          });
        } else {
          data.push({
            aspect: 'spline',
            values,
            text: `Person ${person.id}`,
          });
        }
      });

      return data;
    },

    mapEmotions(emotion) {
      switch (emotion) {
        case 'Neutral':
          return 'Neutral 😐';

        case 'Happy':
          return 'Happy 😊';

        case 'Sad':
          return 'Sad 😞';

        case 'Surprise':
          return 'Surprise 😲';

        case 'Disgust':
          return 'Disguist 🤢';

        default:
          return 'Neutral 😐';
      }
    },

    graphChanged(value) {
      if (value === 'Horizontal') {
        this.criterias = ['Emotions', 'Attention'];
        this.criteriaType = 'Emotions';
      } else if (value === 'Aggregate') {
        this.criterias = ['Attention', 'Happiness', 'Sadness', 'Disgust', 'Surprisement'];
        this.criteriaType = 'Attention';
      }
    },
  },
};
</script>

<style>
.title {
  text-align: center;
  margin-top: 10px;
  margin-bottom: 20px;
}
</style>
