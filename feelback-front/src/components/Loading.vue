<template>
  <div fluid class="fullscreen">
    <div id="spin">
      <img src="../assets/logo.png" width="300" height="300" />
    </div>

    <div class="perc">{{ percentage }}%</div>
    <div class="text">Your video is being processed</div>
  </div>
</template>

<script>
export default {
  data: () => ({
    percentage: 33,
  }),

  mounted() {
    setInterval(() => {
      this.percentage = this.percentage < 100 ? (this.percentage += 1) : 100;
    }, 300);
  },
};
</script>

<style>
.fullscreen {
  width: 100vw;
  height: 100vh;
  background-color: white;
  display: flex;
  justify-content: center;
  align-items: center;
  color: black;
  flex-direction: column;
}

.text {
  font-size: 18px;
  margin-bottom: 50px;
  font-weight: 300;
}

.perc {
  font-size: 32px;
  font-weight: 700;
}

/* Logo animation */
#spin {
  animation-name: grow;
  animation-duration: 3s;
  animation-iteration-count: infinite;
  animation-timing-function: linear;
  -webkit-animation-name: grow;
  -webkit-animation-duration: 3s;
  -webkit-animation-iteration-count: infinite;
  -webkit-animation-timing-function: linear;
  -moz-animation-name: grow;
  -moz-animation-duration: 3s;
  -moz-animation-iteration-count: infinite;
  -moz-animation-timing-function: linear;
  -ms-animation-name: grow;
  -ms-animation-duration: 3s;
  -ms-animation-iteration-count: infinite;
  -ms-animation-timing-function: linear;
  margin-bottom: 20px;
}

@keyframes grow {
  0% {
    -webkit-transform: scale(1);
  }

  50% {
    -webkit-transform: scale(1.2);
  }

  100% {
    -webkit-transform: scale(1);
  }
}

@-webkit-keyframes grow {
  0% {
    -webkit-transform: scale(1);
  }

  50% {
    -webkit-transform: scale(1.2);
  }

  100% {
    -webkit-transform: scale(1);
  }
}

@-moz-keyframes grow {
  0% {
    -moz-transform: scale(1);
  }

  50% {
    -moz-transform: scale(1.2);
  }

  100% {
    -moz-transform: scale(1);
  }
}

@-ms-keyframes grow {
  0% {
    -ms-transform: scale(1);
  }

  50% {
    -ms-transform: scale(1.2);
  }

  100% {
    -ms-transform: scale(1);
  }
}
</style>
