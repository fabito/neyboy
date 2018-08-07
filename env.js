function __delay__(timer) {
    return new Promise(resolve => {
        timer = timer || 2000;
        setTimeout(function () {
            resolve();
        }, timer);
    });
};

function guid() {
  function s4() {
    return Math.floor((1 + Math.random()) * 0x10000)
      .toString(16)
      .substring(1);
  }
  return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
}	

class NeyboyChallenge {

	constructor(id, width=180, height=320) {
		this.id = `${id}`;
		this.iframe = document.createElement('iframe');
		this.iframe.id = this.id;
		this.iframe.src = 'game/mindex.html';
		this.iframe.className = 'game';
		this.iframe.setAttribute('frameborder', 0);
		this.iframe.setAttribute('width', `${width}px`);
		this.iframe.setAttribute('height', `${height}px`);
	}


	static async load(el, numGames) {
		let games = []
		for (let i = 0; i < numGames; i++) {
			let n = new NeyboyChallenge(guid());
			games.push(n.load(el))
		}			
	    return Promise.all(games).then(gameList => {
	    	gameList.reduce(function(obj, x) {obj[x.id] = x; return obj;}, this._registry);
	    	return this._registry;
	    });
	}

	static getGame(id) {
		return NeyboyChallenge._registry[id]
	}

	isReplayButtonActive() {
        const modal = this.runtime.getLayerByName('modal');
        let replay = modal.instances[0];
        if (replay.behavior_insts) {
            return replay.behavior_insts[0].active;
        }
        return false;
	}

	shuffleToasts() {
		let frames = this.contentWindow.data.project[3][20][7][0].frames;
		this.contentWindow.data.project[3][20][7][0].frames = frames.map((a) => [Math.random(),a]).sort((a,b) => a[0]-b[0]).map((a) => a[1]);
	}

	remove() {
		delete NeyboyChallenge._registry[this.id];
		this.iframe.parentNode.removeChild(this.iframe);
		this.runtime = null;
		this.contentWindow = null;
		delete this;
	}

	async load(el) {
		
		await new Promise(resolve => {
			this.iframe.onload = e => {
            	resolve(e);
        	};
			el.appendChild(this.iframe);
		});

		this.contentWindow = this.iframe.contentWindow;
		
	    while (this.contentWindow.cr_getC2Runtime == undefined) {
	    	await __delay__(10);
	    }

	    this.runtime = this.contentWindow.cr_getC2Runtime();

		let flag = this.contentWindow.cr_getC2Runtime !== undefined;

	    while (this.runtime.isloading && this.runtime.loadingprogress <= 0 && this.status() != 0) {
	    	await __delay__(10);
	    }

	    return this;

	}

	pause() {
        this.contentWindow.cr_setSuspended(true);
        return this;
	}

	resume() {
        this.contentWindow.cr_setSuspended(false);
        return this;
	}

	getScore() {
        const score = this.runtime.getLayerByName('Game').instances[4].text || 0;
        return score;			
	}

	dimensions() {
        let {x, y, width, height} = this.iframe.getBoundingClientRect();
        return {x, y, width, height};			
	}

	async getGameLayer() {
    	// this.runtime = this.contentWindow.cr_getC2Runtime();
	    while (this.runtime.getLayerByName('Game') == undefined) {
	    	await __delay__(10);
	    }
	    return this.runtime.getLayerByName('Game');
	}

	status(){
        return this.runtime.getEventVariableByName('isPlaying').data;
	}


	async state(includeSnapshot, format, quality) {
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
                this.contentWindow['cr_onSnapshot'] = function(snapshot) {
                    resolve({score, hiscore, snapshot, status, dimensions, position});
                }
                this.contentWindow.cr_getSnapshot(format, quality);
            } else {
                const snapshot = null;
                resolve({score, hiscore, snapshot, status, dimensions, position});
            }
        })
    }			
}

NeyboyChallenge._registry = {}

var GAMES = {};

function getParameterByName(name, url) {
    if (!url) url = window.location.href;
    name = name.replace(/[\[\]]/g, '\\$&');
    var regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
}	

(async function() { // async function expression used as an IIFE
	let numGames = getParameterByName('n');
	let games = await NeyboyChallenge.load(document.body, parseInt(numGames, 10));
	return games
})().then(games => {
  GAMES = games; 	
  console.log(games);
});
