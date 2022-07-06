<template>
  <zingchart :data="chartConfig" :series="series" />
</template>

<script>
import zingchartVue from 'zingchart-vue';

export default {
  props: {
    title: String,
    data: Object,
    colors: Object,
  },
  data() {
    return {
      chartConfig: {
        theme: 'dark',
        height: '300px',
        type: 'ring',

        backgroundColor: 'none', // This is in the root
        plotarea: {
          backgroundColor: 'transparent',
        },

        legend: {
          x: '80%',
          y: '30%',
        },

        'scale-r': {
          'ref-angle': 270,
        },
        plot: {
          slice: '80',
          tooltip: {
            text: '%t : %v',
          },
          'value-box': {
            text: this.title,
            placement: 'center',
            'font-color': 'White',
            'font-size': 30,
            'font-family': 'Roboto',
            'font-weight': '100',
            rules: [
              {
                rule: '%p != 0',
                visible: false,
              },
            ],
          },
        },
      },
    };
  },
  components: {
    zingchart: zingchartVue,
  },
  computed: {
    series() {
      if (!this.data) {
        return [
          {
            values: [59],
            backgroundColor: '#2196F3',
          },
          {
            backgroundColor: '#F44336',
            values: [55],
          },
        ];
      }
      const values = [];
      Object.entries(this.data).forEach((object) => {
        const [key, value] = object;
        values.push({
          values: [Math.round((value + Number.EPSILON) * 100) / 100],
          backgroundColor: this.colors[key],
          text: key,
        });
      });
      return values;
    },
  },
};
</script>

<style></style>
