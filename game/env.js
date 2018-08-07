function __delay__(timer) {
    return new Promise(resolve => {
        timer = timer || 2000;
        setTimeout(function () {
            resolve();
        }, timer);
    });
};

function getParameterByName(name, url) {
    if (!url) url = window.location.href;
    name = name.replace(/[\[\]]/g, '\\$&');
    var regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
};	

class NeyboyChallenge {

	constructor(ctx, el, width=180, height=320, fullscreen_mode=5) {
		this.ctx = ctx;
		this.data = ctx.data;
		this.data.project[10] = width;
		this.data.project[11] = height;
		// 0 = off, 1 = crop, 2 = scale inner, 3 = scale outer, 4 = letterbox scale, 5 = integer letterbox scale
		this.data.project[12] = fullscreen_mode; 
		// this.audio_to_preload = pm[7];
		this.data.project[7] = []
		// this.preloadSounds = pm[25];
		this.data.project[25] = false;

		// this.data.project[15] = false;
		this.runtime = ctx.cr_createRuntime(el);

		// console.log(this.data.project[15])

		// this.audio_to_preload = pm[7];
		// this.files_subfolder = pm[8];
		// this.pixel_rounding = pm[9];
		// this.aspect_scale = 1.0;
		// this.enableWebGL = pm[13];
		// this.linearSampling = pm[14];
		// this.clearBackground = pm[15];
		// this.versionstr = pm[16];
		// this.useHighDpi = pm[17];
		// this.orientations = pm[20];		// 0 = any, 1 = portrait, 2 = landscape
		// this.autoLockOrientation = (this.orientations > 0);
		// this.pauseOnBlur = pm[22];
		// this.wantFullscreenScalingQuality = pm[23];		// false = low quality, true = high quality
		// this.fullscreenScalingQuality = this.wantFullscreenScalingQuality;
		// this.downscalingQuality = pm[24];	// 0 = low (mips off), 1 = medium (mips on, dense spritesheet), 2 = high (mips on, sparse spritesheet)
		// this.preloadSounds = pm[25];


		// setInterval(()=>{
		// 		console.log(this.dimensions());
		// }, 1000)
	
	}

	isReady(){
		let touchToStartText = this.runtime.getObjectByUID(6);
		if (touchToStartText)
			return touchToStartText.visible;
		else
			return false;
	}

	isOver(){
		let replayButton = this.runtime.getObjectByUID(16);
		if (replayButton && replayButton.behavior_insts)
			return replayButton.behavior_insts[0].useCurrent;
		else
			return false;
	}

	isReplayButtonActive() {
        return this.isOver();
	}

	shuffleToasts() {
		let frames = this.data.project[3][20][7][0].frames;
		this.data.project[3][20][7][0].frames = frames.map((a) => [Math.random(),a]).sort((a,b) => a[0]-b[0]).map((a) => a[1]);
	}

	pause() {
        this.ctx.cr_setSuspended(true);
        return this;
	}

	resume() {
        this.ctx.cr_setSuspended(false);
        return this;
	}

	getScore() {
        const score = this.runtime.getLayerByName('Game').instances[4].text || '0';
        return parseInt(score, 10);			
	}

	dimensions() {
        let {x, y, width, height} = this.runtime.canvas.getBoundingClientRect();

        x = x || 0;
		y = y || 0;
        
        return {x, y, width, height};			
	}

	async getGameLayer() {
	    while (this.runtime.getLayerByName('Game') == undefined) {
	    	await __delay__(10);
	    }
	    return this.runtime.getLayerByName('Game');
	}

	status(){
        return this.runtime.getEventVariableByName('isPlaying').data;
	}

	async state(includeSnapshot=true, format='image/jpeg', quality=30) {
		const gameLayer = await this.getGameLayer();
        return new Promise((resolve, reject) => {
        	
            const dimensions = this.dimensions()
            const score = this.getScore();
            const hiscore = this.runtime.getEventVariableByName('hiscore').data || 0;
            const inst3 = gameLayer.instances[3];
            const position = {
                name: inst3.animTriggerName,
                angle: inst3.angle
            };                
            const status = this.status();

            if (includeSnapshot) {
                this.ctx['cr_onSnapshot'] = function(snapshot) {
                    resolve({score, hiscore, snapshot, status, dimensions, position});
                }
                this.ctx.cr_getSnapshot(format, quality);
            } else {
                const snapshot = null;
                resolve({score, hiscore, snapshot, status, dimensions, position});
            }
        })
    }			
}

