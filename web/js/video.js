import React from 'react'
import {Platform, StyleSheet, View} from 'react-native';
import { UtilComponent } from './util';
import { Player, VolumeMenuButton, ControlBar, PlayToggle, PlaybackRateMenuButton, CurrentTimeDisplay, TimeDivider, DurationDisplay, ProgressControl, RemainingTimeDisplay, FullscreenToggle, BigPlayButton } from 'video-react';
import 'video-react/dist/video-react.css';

class Video extends UtilComponent {
    constructor(props) {
        super(props);
        this.vref = React.createRef();
        this.backoff =1;
        this.backoff_tick = 0;
    }
    componentDidMount() {
        console.log(this.vref);
    }
    componentDidUpdate() {
        console.log(this.vref);
    }
    timejump() {
        if (!this.vref.current) { return; }
        var player = (this.vref.current.getState() || {}).player;
        if (!player) { return; }
        var submin1 = Math.max(this.props.min_time-1, 0);
        var submin2 = Math.max(this.props.min_time-2, 0);
        if (this.props.max_time !== undefined) {
            if ((player.currentTime > this.props.max_time || player.currentTime < submin2)) {
                if (this.backoff_tick == 0) {
                    this.vref.current.play();
                    this.vref.current.seek(submin1);
                    this.pause_expected=1;
                    this.backoff_tick = this.backoff;
                    this.backoff *= 2;
                    this.backoff = Math.min(6, this.backoff);
                } else {
                    this.backoff_tick -= 1;
                    this.pause_expected=1;
                    this.vref.current.pause();
                    if (!this.waiting){
                        this.waiting = true;
                    }
                }
            } else {
                this.vref.current.play();
                this.pause_expected=0;
                this.backoff = 1;
                this.backoff_tick=0;
            }
        }
        setTimeout(() => {if(!this.dying){this.waiting = false; this.timejump()}}, 200);

    }
    render({source, style, autoplay=true, short_controls=true, long_controls=false, onClick, pad=90, min_time, max_time}) {
        return <Player
            playsInline
            style={StyleSheet.flatten([style, {height: '100%', width: '100%', overflow: 'hidden', display: 'inline-block'}])}
            loop={true}
            fluid={false}
            muted
            startTime={min_time && Math.max(min_time-4, 0)}
            autoPlay={autoplay ? true : undefined}
            src={source}
            onPause={event => {if(!this.pause_expected){short_controls ? this.vref.current.play() : onClick();  }}}
            onTimeUpdate={event => this.timejump()}
            ref={this.vref}
        >
            <style></style>
            <BigPlayButton style={{display: 'none'}}/>
            {long_controls && <ControlBar autoHide={false} disableDefaultControls>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
                <PlayToggle />
                <CurrentTimeDisplay />
                <TimeDivider />
                <DurationDisplay/>
                <ProgressControl/>
                <RemainingTimeDisplay/>
                <VolumeMenuButton vertical={true}/>

                <PlaybackRateMenuButton rates={[5, 2, 1, 0.5, 0.1]} />
                <FullscreenToggle/>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
            </ControlBar>}
            {short_controls && <ControlBar autoHide={false} disableDefaultControls>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
                <CurrentTimeDisplay />
                <TimeDivider />
                <DurationDisplay/>
                <div style={{flexGrow: 1}}>&nbsp;</div>
                <PlaybackRateMenuButton rates={[5, 2, 1, 0.5, 0.1]} />
                <FullscreenToggle/>
                <div className="video-react-control" style={{width: '' + pad + 'px'}}>&nbsp;</div>
            </ControlBar>}
        </Player> 
    }
}

export default Video;
